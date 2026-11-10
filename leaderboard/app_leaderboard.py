import datetime as dt
import os

import gradio as gr
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from leaderboard.src.about import (
    ABOUT_TEXT,
    CITATION_INTRO_TEXT,
    CITATION_TEXT,
    EVALUATION_QUEUE_TEXT,
    INTRODUCTION_TEXT,
    TITLE,
)
from leaderboard.src.display.css_html_js import custom_css, theme
from leaderboard.src.display.leaderboard import Leaderboard, SelectColumns
from leaderboard.src.display.utils import (
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    METRIC_COLS,
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
from leaderboard.src.submission.submit import add_new_eval, validate_submission


def restart_space():
    API.restart_space(repo_id=REPO_ID)


INTERVAL = int(os.environ.get("COILBENCH_LEADERBOARD_INTERVAL", "300"))


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
        stratum_label=(
            "Stratum (All = paper headline; tight/medium/loose slice one stratum)"
        ),
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[
                c.name for c in fields(AutoEvalColumn) if c.displayed_by_default
            ],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Select Columns to Display:",
        ),
        search_columns=[
            AutoEvalColumn.model_name.name,
            AutoEvalColumn.submitter.name,
        ],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        highlight_columns=METRIC_COLS,
        label="Leaderboard",
        interactive=False,
    )


# Passed to `app_leaderboard.launch()` (here and in app.py).
LAUNCH_KWARGS = {"css": custom_css, "theme": theme}

app_leaderboard = gr.Blocks()
with app_leaderboard:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem(
            "🏅 Coilstellaration Benchmark", elem_id="benchmark-tab-table", id=0
        ):
            init_leaderboard(LEADERBOARD_FRAMES)

        with gr.TabItem("📝 About", elem_id="about-tab", id=2):
            gr.Markdown(ABOUT_TEXT, elem_classes="markdown-text")
            gr.Markdown(CITATION_INTRO_TEXT, elem_classes="markdown-text")
            gr.Code(
                value=CITATION_TEXT,
                language=None,
                label="BibTeX",
                interactive=False,
            )

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
                                label="Finished evaluations",
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
                                label="Running evaluations",
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
                                label="Pending evaluations",
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
                                label="Failed evaluations",
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

            submit_button = gr.Button("Submit Eval")
            submission_result = gr.Markdown()
            submission_inputs = [
                model_name_textbox,
                revision_name_textbox,
                submitter_textbox,
                onnx_path_textbox,
            ]
            # Format checks run first and are shown on the fields. The button is
            # disabled while the submission runs, and re-enabled however it ends.
            submit_button.click(
                lambda *_: gr.Button(interactive=False),
                submission_inputs,
                submit_button,
                validator=validate_submission,
            ).then(
                add_new_eval,
                submission_inputs,
                submission_result,
            ).then(
                lambda: gr.Button(interactive=True),
                None,
                submit_button,
            )


if os.environ.get("SPACE_ID") is not None:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        restart_space,
        "interval",
        seconds=INTERVAL,
        next_run_time=dt.datetime.now() + dt.timedelta(seconds=INTERVAL),
    )
    scheduler.start()


if __name__ == "__main__":
    app_leaderboard.launch(**LAUNCH_KWARGS)
