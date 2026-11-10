import datetime
import os

import gradio as gr
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from leaderboard.src.about import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    LLM_BENCHMARKS_TEXT,
    TITLE,
)
from leaderboard.src.display.css_html_js import custom_css
from leaderboard.src.display.leaderboard import ColumnFilter, Leaderboard, SelectColumns
from leaderboard.src.display.utils import (
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    STRATUM_VIEWS,
    TRACK_LABELS,
    TRACKS,
    AutoEvalColumn,
    fields,
)
from leaderboard.src.envs import API, EVAL_RESULTS_PATH, REPO_ID, RESULTS_REPO, TOKEN
from leaderboard.src.populate import (
    get_evaluation_queue_df,
    get_leaderboard_frames,
)
from leaderboard.src.submission import requests_dataset
from leaderboard.src.submission.submit import add_new_eval
from huggingface_hub import snapshot_download


def restart_space():
    API.restart_space(repo_id=REPO_ID)


# Submission-form radio labels. Each radio toggles its companion slider on/off.
DETERMINISTIC, STOCHASTIC = "Deterministic", "Stochastic"
FULL_SET, LIMIT_N = "Full set", "Limit to N"
INTERVAL = int(os.environ.get("COILBENCH_LEADERBOARD_INTERVAL", "300"))


def submit_eval(
    model,
    revision,
    onnx_path,
    submitter,
    model_type,
    k_samples,
    instance_mode,
    n_instances,
):
    """Adapt the radio/slider controls to `add_new_eval`'s arguments.

    A disabled slider still submits its last value, so the radio is the source
    of truth: deterministic forces a single sample, full-set forces n=0.
    """
    stochastic = model_type == STOCHASTIC
    n = int(n_instances) if instance_mode == LIMIT_N else -1
    return add_new_eval(
        model, revision, submitter, stochastic, int(k_samples), n, onnx_path
    )


### Space initialisation
try:
    print(EVAL_RESULTS_PATH)
    snapshot_download(
        repo_id=RESULTS_REPO,
        local_dir=EVAL_RESULTS_PATH,
        repo_type="dataset",
        tqdm_class=None,
        etag_timeout=30,
        token=TOKEN,
    )
except Exception:
    restart_space()


LEADERBOARD_FRAMES = get_leaderboard_frames(EVAL_RESULTS_PATH, COLS)

try:
    request_records = list(requests_dataset.iter_requests())
except Exception:
    restart_space()
    request_records = []

(
    finished_eval_queue_df,
    running_eval_queue_df,
    pending_eval_queue_df,
    failed_eval_queue_df,
) = get_evaluation_queue_df(request_records, EVAL_COLS)


def init_leaderboard(frames):
    # A fresh leaderboard with no submissions yet should render an empty table,
    # not crash. Ensure every (track, stratum) frame exists with the expected
    # columns so the component can build.
    frames = dict(frames) if frames else {}
    for track in TRACKS:
        for stratum in STRATUM_VIEWS:
            f = frames.get((track, stratum))
            if f is None or f.empty:
                frames[(track, stratum)] = pd.DataFrame(columns=COLS)
    return Leaderboard(
        value=frames[(TRACKS[0], "All")],
        frames=frames,
        tracks=TRACKS,
        strata=STRATUM_VIEWS,
        track_labels=TRACK_LABELS,
        track_label="Track",
        stratum_label="Stratum (All = paper headline; tight/medium/loose slice one stratum)",
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[
                c.name for c in fields(AutoEvalColumn) if c.displayed_by_default
            ],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Select Columns to Display:",
        ),
        search_columns=[AutoEvalColumn.model.name, AutoEvalColumn.submitter.name],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        filter_columns=[
            ColumnFilter(
                AutoEvalColumn.stochastic.name, type="checkboxgroup", label="Stochastic"
            ),
        ],
        bool_checkboxgroup_label="Hide models",
        interactive=False,
    )


app_leaderboard = gr.Blocks()
with app_leaderboard:
    gr.HTML(f"<style>{custom_css}</style>")
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem(
            "🏅 Coilstellaration Benchmark", elem_id="benchmark-tab-table", id=0
        ):
            init_leaderboard(LEADERBOARD_FRAMES)

        with gr.TabItem("📝 About", elem_id="about-tab", id=2):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")

        with gr.TabItem("🚀 Submit here! ", elem_id="submit-tab", id=3):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")

                with gr.Column():
                    with gr.Accordion(
                        f"✅ Finished Evaluations ({len(finished_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            finished_eval_table = gr.components.Dataframe(
                                value=finished_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
                    with gr.Accordion(
                        f"🔄 Running Evaluation Queue ({len(running_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            running_eval_table = gr.components.Dataframe(
                                value=running_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )

                    with gr.Accordion(
                        f"⏳ Pending Evaluation Queue ({len(pending_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            pending_eval_table = gr.components.Dataframe(
                                value=pending_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
                    with gr.Accordion(
                        f"❌ Failed Evaluations ({len(failed_eval_queue_df)})",
                        open=False,
                    ):
                        with gr.Row():
                            failed_eval_table = gr.components.Dataframe(
                                value=failed_eval_queue_df,
                                headers=EVAL_COLS,
                                datatype=EVAL_TYPES,
                                row_count=5,
                            )
            with gr.Row():
                gr.Markdown(
                    "# ✉️✨ Submit your model here!", elem_classes="markdown-text"
                )

            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(label="Model repo (org/model)")
                    revision_name_textbox = gr.Textbox(
                        label="Revision", placeholder="main"
                    )
                    onnx_path_textbox = gr.Textbox(
                        label="ONNX file path", placeholder="model.onnx"
                    )
                    submitter_textbox = gr.Textbox(label="Submitter / contact")
                with gr.Column():
                    stochastic_radio = gr.Radio(
                        choices=[DETERMINISTIC, STOCHASTIC],
                        value=DETERMINISTIC,
                        label="Model type",
                    )
                    k_samples_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=1,
                        label="K samples",
                        interactive=False,
                    )
                    instance_radio = gr.Radio(
                        choices=[FULL_SET, LIMIT_N],
                        value=FULL_SET,
                        label="Instances per track (larger than the split scores all)",
                    )
                    n_instances_slider = gr.Slider(
                        minimum=1,
                        maximum=1000,
                        step=1,
                        value=10,
                        label="N instances per track",
                        interactive=False,
                    )

                    stochastic_radio.change(
                        lambda mode: gr.update(interactive=mode == STOCHASTIC),
                        inputs=stochastic_radio,
                        outputs=k_samples_slider,
                    )
                    instance_radio.change(
                        lambda mode: gr.update(interactive=mode == LIMIT_N),
                        inputs=instance_radio,
                        outputs=n_instances_slider,
                    )

            submit_button = gr.Button("Submit Eval")
            submission_result = gr.Markdown()
            submit_button.click(
                submit_eval,
                [
                    model_name_textbox,
                    revision_name_textbox,
                    onnx_path_textbox,
                    submitter_textbox,
                    stochastic_radio,
                    k_samples_slider,
                    instance_radio,
                    n_instances_slider,
                ],
                submission_result,
            )

    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                lines=20,
                elem_id="citation-button",
            )

if os.environ.get("SPACE_ID") is not None:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        restart_space,
        "interval",
        seconds=INTERVAL,
        next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=INTERVAL),
    )
    scheduler.start()


if __name__ == "__main__":
    app_leaderboard.launch()
