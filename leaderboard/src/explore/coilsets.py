"""A result's predicted coilsets, one instance at a time, drawn with DESC.

Each result's per-instance scores parquet holds one row per scored instance (a
boundary and a set of engineering requirements), with the predicted coilset as
JSON. A boundary appears once per requirement set, so a row is identified by
its position in the file, not by its boundary id.

Only the coils are drawn for now. The plasma boundary isn't in the parquet; it
can be added once the eval dataset is available here, by drawing its surface
into the same figure before the coils (see `coilstellaration.plot_utils`).
"""

import functools
import logging
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from leaderboard.src.display.css_html_js import STEEL_BLUE
from leaderboard.src.envs import EVAL_RESULTS_PATH, RESULTS_REPO, TOKEN

logger = logging.getLogger(__name__)

# The engineering requirements, as named in the parquet's `violation/` columns,
# with the label shown in the instance summary.
REQUIREMENTS = {
    "min_normalized_coil_to_coil_distance": "Coil-to-coil distance",
    "min_normalized_coil_to_plasma_distance": "Coil-to-plasma distance",
    "max_normalized_coil_curvature": "Coil curvature",
    "max_normalized_field_error": "Field error",
}
_SUMMARY_COLUMNS = ["track", "stratum", "boundary_id", "score", "strictly_feasible"] + [
    f"violation/{name}" for name in REQUIREMENTS
]


@functools.lru_cache(maxsize=8)
def parquet_path(raw_scores_file: str) -> str:
    """The local path of a result's scores parquet, downloaded on first use.

    The leaderboard downloads only the JSON summaries at start-up; each parquet
    (tens of MB) is fetched only when someone opens that result here.
    """
    try:
        return hf_hub_download(
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            filename=raw_scores_file,
            local_dir=EVAL_RESULTS_PATH,
            token=TOKEN,
        )
    except Exception:
        # Offline, or the Hub is down: use a copy that is already here, if any.
        local = Path(EVAL_RESULTS_PATH) / raw_scores_file
        if local.exists():
            logger.warning("Using the local copy of %s", raw_scores_file)
            return str(local)
        raise


def instances(path: str, track: str) -> pd.DataFrame:
    """The track's instances, best score first, indexed by their row in the file.

    Holds the summary columns only; the coilsets are read one at a time by
    `load_coilset`.
    """
    df = pd.read_parquet(path, columns=_SUMMARY_COLUMNS)
    df = df[df["track"] == track]
    return df.sort_values("score", ascending=False, kind="stable")


def instance_label(row_index: int, row: pd.Series) -> str:
    """The instance's entry in the picker: score, feasibility, stratum and boundary."""
    feasible = "feasible" if row["strictly_feasible"] else "infeasible"
    return (
        f"{row['score']:.3f} · {feasible} · {row['stratum']} · "
        f"{row['boundary_id']} (#{row_index})"
    )


def load_coilset(path: str, row_index: int):
    """The predicted coilset in row `row_index`, or `None` if inference failed."""
    from coilstellaration import types

    table = pq.read_table(path, columns=["coilset_json"])
    coilset_json = table.column("coilset_json")[row_index].as_py()
    if coilset_json is None:
        return None
    return types.Coilset.model_validate_json(coilset_json)


def summary_markdown(row: pd.Series) -> str:
    """The instance's score and how far it misses each requirement."""
    feasible = (
        "strictly feasible" if row["strictly_feasible"] else "not strictly feasible"
    )
    lines = [
        f"**Boundary** `{row['boundary_id']}`, {row['stratum']} stratum  ",
        f"**Score** {row['score']:.3f}, {feasible}",
        "",
        "| Requirement | Violation |",
        "| --- | --- |",
    ]
    for name, label in REQUIREMENTS.items():
        value = row[f"violation/{name}"]
        shown = "met" if value <= 0 else f"{value:.3g}"
        lines.append(f"| {label} | {shown} |")
    lines += [
        "",
        "Violations are normalised by the requirement; 0 means it is met.",
    ]
    return "\n".join(lines)


def figure(coilset) -> go.Figure:
    """The full coilset (every symmetry copy of the predicted coils) in 3D.

    Steel-blue reads on both the white and the navy page, so the figure needs no
    dark-mode variant. The axes are hidden: the coils' shape is the point.
    """
    # Imported here: DESC pulls in JAX, which takes seconds to load, and only
    # this view needs it.
    from desc.plotting import plot_coils

    from coilstellaration import coilset_utils

    desc_coilset = coilset_utils.coilstellaration_to_desc(
        coilset, check_intersection=False
    )
    with warnings.catch_warnings():
        # Predicted coils often nearly touch; the score already accounts for it.
        warnings.simplefilter("ignore", UserWarning)
        fig = plot_coils(desc_coilset, color=STEEL_BLUE, lw=4)
    hidden = {"visible": False}
    fig.update_layout(
        height=560,
        autosize=True,
        width=None,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene={
            "xaxis": hidden,
            "yaxis": hidden,
            "zaxis": hidden,
            "aspectmode": "data",
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig
