import datetime as dt
import os

import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler

from leaderboard.src.about import (
    ABOUT_TEXT,
    CHANGELOG_TEXT,
    CITATION_INTRO_TEXT,
    CITATION_TEXT,
    DATASET_URL,
    EVALUATION_QUEUE_TEXT,
    FOOTER_TEXT,
    INTRODUCTION_TEXT,
    ORG_URL,
    TITLE_CONT,
    TITLE_LEAD,
)
from leaderboard.src.display.css_html_js import custom_css, header_html, theme
from leaderboard.src.display.leaderboard import Leaderboard, SelectColumns
from leaderboard.src.display.utils import (
    DEFAULT_VIEW,
    EVAL_TYPES,
    INTEGER_COLS,
    METRIC_COLS,
    STRATUM_VIEWS,
    TRACK_LABELS,
    TRACKS,
    AutoEvalColumn,
    fields,
)
from leaderboard.src.explore import tab as explore
from leaderboard.src.live import QUEUE_STATUSES, LiveData
from leaderboard.src.submission.submit import (
    NO_EVAL_TRAINING_DECLARATION,
    add_new_eval,
    validate_submission,
)

# Seconds between refreshes of the results and the evaluation queue.
INTERVAL = int(os.environ.get("COILBENCH_LEADERBOARD_INTERVAL", "300"))

# The data on show, refreshed in the background (see the end of this file).
# Event handlers read `LIVE.current` each time they run, so every page load gets
# the latest refresh without restarting the Space.
LIVE = LiveData()
LIVE.refresh()


def init_leaderboard(reload_on):
    """The leaderboard table, showing `LIVE`'s frames and redrawn on `reload_on`."""
    return Leaderboard(
        value=LIVE.current.frames[DEFAULT_VIEW],
        frames=lambda: LIVE.current.frames,
        reload_on=reload_on,
        tracks=TRACKS,
        strata=STRATUM_VIEWS,
        default_view=DEFAULT_VIEW,
        track_labels=TRACK_LABELS,
        track_label="Track",
        stratum_label="Stratum",
        datatype=[c.type for c in fields(AutoEvalColumn)],
        select_columns=SelectColumns(
            default_selection=[
                c.name for c in fields(AutoEvalColumn) if c.displayed_by_default
            ],
            cant_deselect=[c.name for c in fields(AutoEvalColumn) if c.never_hidden],
            label="Columns",
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


QUEUE_ICONS = {"Running": "🔄", "Pending": "⏳", "Finished": "✅", "Failed": "❌"}


def queue_title(status: str, count: int) -> str:
    return f"{QUEUE_ICONS[status]} {status} ({count})"


def show_queue():
    """The evaluation queue's accordion titles, then its tables, as of now."""
    queue = LIVE.current.queue
    return [
        *(gr.Accordion(label=queue_title(s, len(queue[s]))) for s in QUEUE_STATUSES),
        *(queue[s] for s in QUEUE_STATUSES),
    ]


TAB_IDS = ("leaderboard", "explore", "about", "submit")


# Passed to `app_leaderboard.launch()` (here and in app.py).
LAUNCH_KWARGS = {"css": custom_css, "theme": theme}

# Every hour, delete the files Gradio has cached (the leaderboard's CSV downloads)
# that are over a day old. A component's initial value is never deleted.
app_leaderboard = gr.Blocks(
    title="Coilstellaration benchmark | Proxima Fusion",
    delete_cache=(3600, 86400),
)
with app_leaderboard:
    gr.HTML(header_html(TITLE_LEAD, TITLE_CONT, lead_url=DATASET_URL, logo_url=ORG_URL))
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("Leaderboard", elem_id="benchmark-tab-table", id="leaderboard"):
            init_leaderboard(reload_on=app_leaderboard.load)

        # Loaded lazily: nothing in it is computed until the tab is opened.
        with gr.TabItem("Explore", elem_id="explore-tab", id="explore") as explore_tab:
            explore_view = explore.render(lambda: LIVE.current.records)

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
                    submission_result = gr.Markdown(
                        label="Submission result", elem_classes="markdown-text"
                    )
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
            # Only the submission itself is an API endpoint, `/submit`; API
            # callers skip the validator, so `add_new_eval` repeats its checks.
            submit_button.click(
                lambda *_: gr.Button(interactive=False),
                submission_inputs,
                submit_button,
                validator=validate_submission,
                api_visibility="private",
            ).then(
                add_new_eval,
                submission_inputs,
                submission_result,
                api_name="submit",
            ).then(
                lambda: gr.Button(interactive=True),
                None,
                submit_button,
                api_visibility="private",
            )

            gr.Markdown("## Evaluation queue", elem_classes="markdown-text")
            queue_accordions, queue_tables = [], []
            for status in QUEUE_STATUSES:
                frame = LIVE.current.queue[status]
                with gr.Accordion(
                    queue_title(status, len(frame)), open=False
                ) as accordion:
                    table = gr.Dataframe(
                        value=frame,
                        label=f"{status} evaluations",
                        show_label=False,
                        datatype=EVAL_TYPES,
                        row_count=5,
                        interactive=False,
                    )
                queue_accordions.append(accordion)
                queue_tables.append(table)

    gr.Markdown(FOOTER_TEXT, elem_id="pf-footer")

    explore_tab.select(
        explore_view.draw_pareto,
        explore_view.pareto_inputs,
        explore_view.pareto_outputs,
        api_visibility="private",
    )

    def select_tab_from_url(request: gr.Request):
        """Open the tab named by the `tab` query parameter, if it names one.

        Selecting a tab this way doesn't fire its select event, so the Explore
        tab's Pareto front is drawn here too, for the controls' default values:
        the page has only just loaded, so nobody has changed them. The controls
        aren't passed as inputs, because inputs from a tab that isn't shown yet
        stall the page's first render by tens of seconds.
        """
        tab = request.query_params.get("tab")
        if tab not in TAB_IDS:
            return gr.skip(), gr.skip(), gr.skip()
        if tab == "explore":
            pareto = explore_view.draw_pareto(*explore_view.pareto_defaults)
            return gr.Tabs(selected=tab), *pareto
        return gr.Tabs(selected=tab), gr.skip(), gr.skip()

    app_leaderboard.load(
        show_queue,
        None,
        [*queue_accordions, *queue_tables],
        api_visibility="private",
    )

    # `?tab=explore`, `?tab=about` or `?tab=submit` in the URL opens that tab.
    app_leaderboard.load(
        select_tab_from_url,
        None,
        [tabs, *explore_view.pareto_outputs],
        api_visibility="private",
    )

scheduler = BackgroundScheduler()
scheduler.add_job(
    LIVE.refresh,
    "interval",
    seconds=INTERVAL,
    next_run_time=dt.datetime.now() + dt.timedelta(seconds=INTERVAL),
)
scheduler.start()


if __name__ == "__main__":
    app_leaderboard.launch(**LAUNCH_KWARGS)
