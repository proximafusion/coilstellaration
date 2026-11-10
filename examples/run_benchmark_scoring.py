"""Local quick-check scoring of one ML model checkpoint on the public eval set.

Official benchmark numbers come from the Hugging Face Space. Run with `--help` for the
flags; the README's end-to-end walkthrough describes the flags and outputs.
"""

import contextlib
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import textwrap
import warnings
from collections.abc import Iterator, Sequence
from typing import Literal, get_args

import click
import datasets
import matplotlib.pyplot as plt
import pandas as pd

from coilstellaration import (
    coilset_utils,
    data_generation_tasks,
    data_utils,
    desc_utils,
    metrics_utils_v2,
    paths,
    types,
)
from coilstellaration.benchmark import (
    scoring,
    scoring_types,
)
from coilstellaration.machine_learning import (
    model_definition,
)
from coilstellaration.machine_learning import (
    utils as ml_utils,
)

logger = logging.getLogger(__name__)

Stratum = Literal["loose", "medium", "tight"]
STRATA: tuple[Stratum, ...] = get_args(Stratum)
NFP_COLUMN = "boundary.n_field_periods"
STRATUM_COLUMN = "stratum"


@click.command(name="run_benchmark_scoring")
@click.option("--checkpoint-id", required=True)
@click.option(
    "--model-kind",
    default="mlp",
    show_default=True,
    type=click.Choice(get_args(types.AnyModelLiteral)),
)
@click.option(
    "--track",
    default="fixed_shape",
    show_default=True,
    type=click.Choice(["fixed_shape", "variable_shape"]),
)
@click.option(
    "--stratum", default="tight", show_default=True, type=click.Choice([*STRATA, "all"])
)
@click.option(
    "--n",
    default=4,
    show_default=True,
    help="Instances per stratum (per NFP x coil-count cell with --stratify-nfp-nc). "
    "0 means all.",
)
@click.option("--stratify-nfp-nc", is_flag=True)
@click.option("--instance-timeout-s", default=1800.0, show_default=True)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Sampling seed; only affects --stratify-nfp-nc.",
)
@click.option(
    "--desc-refine-from",
    type=click.Choice(["model", "regcoil"]),
    default="model",
    help="Starting coilset, refined with DESC when --desc-maxiters is above 0.",
)
@click.option(
    "--desc-maxiters",
    type=int,
    multiple=True,
    default=(0,),
    show_default=True,
    help="DESC iterations; repeat to score several in one worker per instance, which "
    "then compiles once. 0 scores the starting coilset as is, like the HF Space.",
)
def cli(
    checkpoint_id: str,
    model_kind: types.AnyModelLiteral,
    track: Literal["fixed_shape", "variable_shape"],
    stratum: Stratum | Literal["all"],
    n: int,
    stratify_nfp_nc: bool,
    instance_timeout_s: float,
    seed: int,
    desc_refine_from: Literal["model", "regcoil"],
    desc_maxiters: tuple[int, ...],
) -> pd.DataFrame:
    """Local quick-check scoring (official numbers: HF Space)."""
    return main(
        checkpoint_id=checkpoint_id,
        model_kind=model_kind,
        track=track,
        stratum=stratum,
        n=n,
        stratify_nfp_nc=stratify_nfp_nc,
        instance_timeout_s=instance_timeout_s,
        seed=seed,
        desc_refine_from=desc_refine_from,
        desc_maxiters=desc_maxiters,
    )


def main(
    checkpoint_id: str,
    model_kind: types.AnyModelLiteral,
    track: Literal["fixed_shape", "variable_shape"],
    stratum: Stratum | Literal["all"],
    n: int,
    stratify_nfp_nc: bool,
    instance_timeout_s: float,
    seed: int,
    desc_refine_from: Literal["model", "regcoil"],
    desc_maxiters: Sequence[int],
) -> pd.DataFrame:
    """Local quick-check scoring (official numbers: HF Space).

    Returns:
        Per-instance rows (see `instance_rows`) of every maxiter, labelled in a
        `desc_maxiter` column.
    """
    plt.switch_backend("Agg")

    settings = scoring_types.ScoringSettings()

    df = load_rows(
        stratum=stratum,
        track=track,
        stratify_nfp_nc=stratify_nfp_nc,
        n=n,
        seed=seed,
    )
    eval_dataset = data_utils.load_dataset(df)

    match desc_refine_from:
        case "regcoil":
            coilsets_dataset = datasets.load_dataset(
                "proxima-fusion/coilstellaration", name="coilsets", split="train"
            )
            assert type(coilsets_dataset) is datasets.Dataset
            coilsets_df = coilsets_dataset.to_pandas(batched=False)
            assert type(coilsets_df) is pd.DataFrame
            coilsets_json = dict(
                zip(coilsets_df["id"], coilsets_df["json"], strict=True)
            )
            predictions = [
                eval_data.model_copy(
                    update={
                        "predicted_coilset": types.Coilset.model_validate_json(
                            coilsets_json[coilset_id]
                        )
                    }
                )
                for eval_data, coilset_id in zip(
                    eval_dataset, df["regcoil_coilset_id"], strict=True
                )
            ]
        case "model":
            checkpoint = ml_utils.read_checkpoint_from_id(model_kind, checkpoint_id)
            model = model_definition.read_model_from_checkpoint(checkpoint)
            logger.info("Predicting %d coilsets...", len(eval_dataset))
            predictions = model_definition.predict_coilsets(model, eval_dataset)
        case _:
            raise ValueError(f"{desc_refine_from} is not supported.")

    desc_equilibria = load_desc_equilibria(df)
    scores_at_maxiter: list[dict[int, scoring_types.InstanceScore]] = []
    for i, (eval_data, desc_equilibrium, requirements_json) in enumerate(
        zip(predictions, desc_equilibria, df["json_requirements"], strict=True)
    ):
        scores_at_maxiter.append(
            score_instance(
                eval_data,
                desc_equilibrium,
                requirements_json,
                maxiters=desc_maxiters,
                timeout_s=instance_timeout_s,
                settings=settings,
            )
        )
        logger.info("Scored %d / %d", i + 1, len(predictions))

    frames = []
    for maxiter in sorted(set(desc_maxiters)):
        instance_scores = [scores[maxiter] for scores in scores_at_maxiter]
        scores_df = instance_rows(df, instance_scores)
        cid = f"{checkpoint_id}_desc_from_{desc_refine_from}_maxiter{maxiter}"
        scores_csv = paths.OUTPUTS_PATH / f"quickcheck_scores_{cid}.csv"
        scores_df.to_csv(scores_csv, index=False)
        logger.info("Wrote per-instance scores to %s", scores_csv)

        plot_soft_feasibility(
            scores_df,
            paths.OUTPUTS_PATH / f"quickcheck_soft_feasibility_{cid}.png",
            title=f"Quick check soft feasibility - {cid}",
        )

        summary_df = summarize_by_stratum(scores_df, instance_scores, settings)
        summary_df.to_csv(paths.OUTPUTS_PATH / f"quickcheck_summary_{cid}.csv")
        nfp_nc_df = summarize_by_nfp_nc(scores_df)
        nfp_nc_df.to_csv(paths.OUTPUTS_PATH / f"quickcheck_summary_nfp_nc_{cid}.csv")
        logger.info(
            "Quick-check scores for %s (n=%d); official numbers come from the HF "
            "Space:",
            cid,
            len(instance_scores),
        )
        print(summary_df)
        print(nfp_nc_df)
        frames.append(scores_df.assign(desc_maxiter=maxiter))
    return pd.concat(frames, ignore_index=True)


def check_args(argv: Sequence[str]) -> None:
    """Raises `click.UsageError` if `argv` is not valid for `cli`."""
    with cli.make_context(cli.name, list(argv)):
        pass


def load_rows(
    stratum: Stratum | Literal["all"],
    track: Literal["fixed_shape", "variable_shape"],
    stratify_nfp_nc: bool,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Eval rows of the selected strata, with the stratum in `STRATUM_COLUMN`.

    A boundary can appear in several strata (with different requirements), so rows are
    identified by position, not by boundary ID.
    """
    strata = STRATA if stratum == "all" else (stratum,)
    frames = []
    for stratum_name in strata:
        df = data_utils.load_benchmark_dataframe(
            track=track, stratum=stratum_name, split="eval"
        )
        if stratify_nfp_nc and n > 0:
            df = data_utils.sample_stratified_rows(df, n_per_cell=n, seed=seed)
        elif n > 0:
            df = df.head(n)
        frames.append(df.assign(**{STRATUM_COLUMN: stratum_name}))
    return pd.concat(frames, ignore_index=True)


def load_desc_equilibria(df: pd.DataFrame) -> list[types.DescEquilibrium]:
    """The dataset's DESC equilibrium of each row.

    Raises:
        ValueError: If a row has no equilibrium.
    """
    equilibrium_ids = df[data_utils.DESC_EQUILIBRIUM_ID_COLUMN]
    n_missing = int(equilibrium_ids.isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} of {len(df)} rows have no "
            f"`{data_utils.DESC_EQUILIBRIUM_ID_COLUMN}`."
        )
    equilibria = data_utils.load_desc_equilibria_by_ids(equilibrium_ids.tolist())
    return [equilibria[equilibrium_id] for equilibrium_id in equilibrium_ids]


def instance_rows(
    df: pd.DataFrame, instance_scores: Sequence[scoring_types.InstanceScore]
) -> pd.DataFrame:
    """Per-instance rows: ID, stratum, NFP, coil count and DESC coilset ID of the
    matching row, and scores.

    `instance_scores[i]` belongs to the `i`-th row of `df`.
    """
    context = df[
        [STRATUM_COLUMN, NFP_COLUMN, data_utils.N_COILS_COLUMN, "desc_coilset_id"]
    ].to_dict("records")
    return pd.DataFrame(
        [
            {
                "boundary_id": score.boundary_id,
                **row,
                **scoring.instance_score_to_row(score),
            }
            for row, score in zip(context, instance_scores, strict=True)
        ]
    )


def summarize_by_stratum(
    scores_df: pd.DataFrame,
    instance_scores: Sequence[scoring_types.InstanceScore],
    settings: scoring_types.ScoringSettings,
) -> pd.DataFrame:
    """Benchmark summaries over all instances (column `all`) and per stratum.

    `instance_scores[i]` belongs to the `i`-th row of `scores_df`.
    """
    summaries = {
        "all": pd.Series(scoring.score_benchmark(instance_scores, settings).summaries)
    }
    strata = scores_df[STRATUM_COLUMN].tolist()
    for stratum in sorted(set(strata)):
        subset = [
            score
            for score, score_stratum in zip(instance_scores, strata, strict=True)
            if score_stratum == stratum
        ]
        summaries[stratum] = pd.Series(
            scoring.score_benchmark(subset, settings).summaries
        )
    return pd.DataFrame(summaries)


def summarize_by_nfp_nc(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Mean score, strict-feasibility rate and count per (NFP, coils per half
    period)."""
    return (
        scores_df.groupby([NFP_COLUMN, data_utils.N_COILS_COLUMN])
        .agg(
            mean_score=("score", "mean"),
            strict_feasibility=("strictly_feasible", "mean"),
            n=("score", "size"),
        )
        .reset_index()
    )


def parse_dataset_requirements(requirements_json: str) -> types.Requirements:
    """Parse a dataset requirements record for the DESC refinement."""
    return types.Requirements.model_validate(
        {"regcoil_target_option": "normalized_field_error"}
        | json.loads(requirements_json)
    )


def score_instance(
    eval_data: types.EvalData,
    desc_equilibrium: types.DescEquilibrium,
    requirements_json: str,
    maxiters: Sequence[int],
    timeout_s: float,
    settings: scoring_types.ScoringSettings,
) -> dict[int, scoring_types.InstanceScore]:
    """Score one prediction after DESC refinement with each of `maxiters`, in one worker
    subprocess; zero score for every maxiter not reached on timeout or error.

    Args:
        desc_equilibrium: The equilibrium to refine the prediction and evaluate the
            metrics on; the dataset's targets were computed on the dataset's
            equilibrium of the instance.
        requirements_json: The dataset requirements record to refine the prediction
            with DESC for.
        timeout_s: Covers the DESC refinements of all `maxiters`.

    Returns:
        The score of each maxiter.
    """
    assert eval_data.predicted_coilset is not None
    # The worker is a fresh interpreter, so it does not inherit source roots that were
    # added to `sys.path` at runtime.
    python_path = os.pathsep.join(sys.path)
    with tempfile.TemporaryDirectory() as tmp:
        instance_path = pathlib.Path(tmp, "instance.json")
        metrics_path = pathlib.Path(tmp, "metrics.json")
        worker_path = pathlib.Path(tmp, "worker.py")
        instance_path.write_text(
            json.dumps(
                {
                    "desc_equilibrium_json": desc_equilibrium.model_dump_json(),
                    "coilset_json": eval_data.predicted_coilset.model_dump_json(),
                    "requirements_json": requirements_json,
                    "maxiters": list(maxiters),
                }
            )
        )
        worker_path.write_text(
            textwrap.dedent(
                """
            import json
            import logging
            import os
            import pathlib
            import sys

            import jax
            from coilstellaration.examples import (
                run_benchmark_scoring
            )

            if __name__ == "__main__":
                logging.basicConfig(level=logging.INFO, force=True)
                for noisy in ("storage.google_cloud", "httpx", "httpx2", "absl"):
                    logging.getLogger(noisy).setLevel(logging.ERROR)
                # Every instance gets a new worker, which would otherwise recompile
                # from scratch; the disk cache skips XLA's backend compilation, though
                # not tracing.
                jax.config.update(
                    "jax_compilation_cache_dir",
                    str(pathlib.Path.home() / ".cache" / "coilstellaration_jax"),
                )
                jax.config.update("jax_persistent_cache_min_compile_time_secs", 1e-3)
                jax.config.update("jax_persistent_cache_min_entry_size_bytes", 100)
                instance_path, metrics_path = map(pathlib.Path, sys.argv[1:])
                metrics = {}
                for maxiter, metrics_at_maxiter in (
                    run_benchmark_scoring.evaluate_instance(
                        **json.loads(instance_path.read_text())
                    )
                ):
                    # Written after every maxiter so that a timeout keeps the ones
                    # already done; replaced atomically so a kill mid-write cannot
                    # leave a truncated file.
                    metrics[maxiter] = metrics_at_maxiter.model_dump_json()
                    partial_path = metrics_path.with_suffix(".partial")
                    partial_path.write_text(json.dumps(metrics))
                    os.replace(partial_path, metrics_path)
            """
            )
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(worker_path),
                str(instance_path),
                str(metrics_path),
            ],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": python_path},
        )
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Instance %s: no result after %s s", eval_data.boundary_id, timeout_s
            )
        finally:
            # Also reaps helper processes XLA may have left behind.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        achieved = (
            {
                int(maxiter): types.RequirementMetrics.model_validate_json(metrics)
                for maxiter, metrics in json.loads(metrics_path.read_text()).items()
            }
            if metrics_path.exists()
            else {}
        )
    missing = sorted(set(maxiters) - achieved.keys())
    if missing:
        logger.warning(
            "Instance %s: no result for maxiters %s, worker exited with code %d",
            eval_data.boundary_id,
            missing,
            process.returncode,
        )
    return {
        maxiter: scoring.score_eval_data(eval_data, achieved[maxiter], settings)
        if maxiter in achieved
        else scoring.zero_instance_score(eval_data.boundary_id)
        for maxiter in maxiters
    }


def score_instance_from_boundary(
    eval_data: types.EvalData,
    requirements_json: str,
    maxiters: Sequence[int],
    timeout_s: float,
    settings: scoring_types.ScoringSettings,
) -> dict[int, scoring_types.InstanceScore]:
    """`score_instance` on an equilibrium built from `eval_data.boundary` alone.

    Its minor radius and axis can differ from the dataset's equilibrium, so even the
    reference coilset may miss its targets.
    """
    return score_instance(
        eval_data,
        desc_utils.desc_equilibrium_from_boundary(eval_data.boundary),
        requirements_json,
        maxiters=maxiters,
        timeout_s=timeout_s,
        settings=settings,
    )


def evaluate_instance(
    desc_equilibrium_json: str,
    coilset_json: str,
    requirements_json: str,
    maxiters: Sequence[int],
) -> Iterator[tuple[int, types.RequirementMetrics]]:
    """Achieved metrics of one instance at each maxiter, in increasing order, run in the
    worker subprocess of `score_instance`.

    Every maxiter refines the starting coilset from scratch, so its metrics are the
    same as in a run with only that maxiter.

    See `score_instance` for the arguments.
    """
    with warnings.catch_warnings(action="ignore"):
        requirements = parse_dataset_requirements(requirements_json)
        eq = desc_utils.desc_equilibrium_to_desc_object(
            types.DescEquilibrium.model_validate_json(desc_equilibrium_json)
        )
        start_coilset = types.Coilset.model_validate_json(coilset_json)
        scaled_start_coilset = (
            data_generation_tasks.scale_coil_currents_to_B_0_equals_1_T(
                eq, start_coilset
            )
        )
        for maxiter in sorted(set(maxiters)):
            coilset = (
                start_coilset
                if maxiter == 0
                else data_generation_tasks.optimize_coilset_using_desc(
                    eq=eq,
                    coilset=scaled_start_coilset,
                    requirements=requirements,
                    settings=types.DescOptimizerSettings(maxiter=maxiter),
                ).coilset
            )
            metrics = metrics_utils_v2.evaluate_coilset_metrics(
                eq=eq,
                coilset=coilset_utils.coilstellaration_to_desc(coilset),
            )
            yield maxiter, data_utils.metrics_to_requirement_metrics(metrics)


def plot_soft_feasibility(
    scores_df: pd.DataFrame, output_path: pathlib.Path, title: str
) -> None:
    fields = list(scoring_types.REQUIREMENT_METRIC_DIRECTIONS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, field in zip(axes[:4], fields, strict=True):
        ax.hist(
            scores_df[f"soft_score/{field}"].to_numpy(),
            bins=20,
            range=(0.0, 1.0),
        )
        ax.set_title(field)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("soft feasibility")
    axes[4].hist(scores_df["score"].to_numpy(), bins=20, range=(0.0, 1.0), color="C2")
    axes[4].set_title("aggregate (geometric mean)")
    axes[4].set_xlim(-0.05, 1.05)
    axes[4].set_xlabel("score")
    axes[5].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    for noisy in ("storage.google_cloud", "httpx", "httpx2", "absl"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    cli()
