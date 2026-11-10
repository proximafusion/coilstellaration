import datetime as dt
import os

import gradio as gr
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from leaderboard.src.about import (
    ABOUT_TEXT,
    CHANGELOG_TEXT,
    CITATION_INTRO_TEXT,
    CITATION_TEXT,
    EVALUATION_QUEUE_TEXT,
    FOOTER_TEXT,
    INTRODUCTION_TEXT,
    TITLE_CONT,
    TITLE_LEAD,
)
from leaderboard.src.display.css_html_js import custom_css, header_html, theme
from leaderboard.src.display.leaderboard import Leaderboard, SelectColumns
from leaderboard.src.display.utils import (
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    INTEGER_COLS,
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
from leaderboard.src.submission.submit import (
    NO_EVAL_TRAINING_DECLARATION,
    add_new_eval,
    validate_submission,
)


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
        # Only the JSON summaries; the parquet and ONNX files aren't displayed.
        allow_patterns="*.json",
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
        stratum_label="Stratum",
        stratum_info="All pools the three strata (the paper's headline view).",
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[
                c.name for c in fields(AutoEvalColumn) if c.displayed_by_default
            ],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Columns",
            info="Add or remove columns.",
        ),
        search_columns=[
            AutoEvalColumn.model.name,
            AutoEvalColumn.user_name.name,
        ],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        highlight_columns=METRIC_COLS,
        integer_columns=INTEGER_COLS,
        label="Leaderboard",
        download_name="coilstellaration-leaderboard",
        interactive=False,
    )


TAB_IDS = ("leaderboard", "about", "submit")


def select_tab_from_url(request: gr.Request):
    """Open the tab named by the `tab` query parameter, if it names one."""
    tab = request.query_params.get("tab")
    return gr.Tabs(selected=tab) if tab in TAB_IDS else gr.skip()


# Passed to `app_leaderboard.launch()` (here and in app.py).
LAUNCH_KWARGS = {"css": custom_css, "theme": theme}

app_leaderboard = gr.Blocks(title="Coilstellaration benchmark | Proxima Fusion")
with app_leaderboard:
    gr.HTML(header_html(TITLE_LEAD, TITLE_CONT))
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("Leaderboard", elem_id="benchmark-tab-table", id="leaderboard"):
            init_leaderboard(LEADERBOARD_FRAMES)

        with gr.TabItem("About", elem_id="about-tab", id="about"):
            gr.Markdown(ABOUT_TEXT, elem_classes="markdown-text")
            gr.Markdown(CITATION_INTRO_TEXT, elem_classes="markdown-text")
            gr.Code(
                value=CITATION_TEXT,
                language=None,
                label="BibTeX",
                interactive=False,
            )
            gr.Markdown(CHANGELOG_TEXT, elem_classes="markdown-text")

        with gr.TabItem("Submit", elem_id="submit-tab", id="submit"):
            # The form on the left, how submission works on the right.
            with gr.Row():
                with gr.Column(scale=3):
                    model_name_textbox = gr.Textbox(
                        label="Model repo",
                        placeholder="org/model",
                        info="A public Hugging Face model repo.",
                    )
                    revision_name_textbox = gr.Textbox(
                        label="Revision",
                        placeholder="main",
                        info="Branch, tag or commit. It is pinned to a commit on "
                        "submission.",
                    )
                    onnx_path_textbox = gr.Textbox(
                        label="ONNX file path",
                        placeholder="model.onnx",
                        info="Path of the ONNX file within the repo.",
                    )
                    submitter_textbox = gr.Textbox(
                        label="Hugging Face username",
                        info="Shown on the leaderboard as the user, linked to "
                        "your Hugging Face profile.",
                    )
                    no_eval_training_checkbox = gr.Checkbox(
                        label=NO_EVAL_TRAINING_DECLARATION,
                        value=False,
                    )
                    submit_button = gr.Button("Submit model", variant="primary")
                    submission_result = gr.Markdown(elem_classes="markdown-text")
                with gr.Column(scale=2):
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")

            submission_inputs = [
                model_name_textbox,
                revision_name_textbox,
                submitter_textbox,
                onnx_path_textbox,
                no_eval_training_checkbox,
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

            gr.Markdown("## Evaluation queue", elem_classes="markdown-text")
            for icon, title, frame in [
                ("🔄", "Running", running_eval_queue_df),
                ("⏳", "Pending", pending_eval_queue_df),
                ("✅", "Finished", finished_eval_queue_df),
                ("❌", "Failed", failed_eval_queue_df),
            ]:
                with gr.Accordion(f"{icon} {title} ({len(frame)})", open=False):
                    gr.Dataframe(
                        value=frame,
                        label=f"{title} evaluations",
                        show_label=False,
                        datatype=EVAL_TYPES,
                        row_count=5,
                        interactive=False,
                    )

    gr.Markdown(FOOTER_TEXT, elem_id="pf-footer")

    # `?tab=about` or `?tab=submit` in the URL opens that tab.
    app_leaderboard.load(select_tab_from_url, None, tabs)


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
