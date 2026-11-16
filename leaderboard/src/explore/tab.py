"""The Explore tab's layout and events.

Nothing here runs until someone opens one of its sub-tabs: a result's coilsets
are downloaded when the Coilsets sub-tab is first opened, and Netron is framed
only when the Model graph sub-tab is. Gradio also mounts a tab's components only when
the tab is first shown.

The results come from a function, called on every event, so the tab follows data
that is refreshed while the app runs. The Coilsets and Model graph pickers are
filled with the current results when their sub-tab is first opened.

The events that load a sub-tab on first visit are UI-only (`api_visibility=
"private"`). The rest are published as API endpoints: `/coilset_instances`,
`/coilset` and `/model_graph`.
"""

import functools
import logging
from collections.abc import Callable
from typing import cast

import gradio as gr

from leaderboard.src.display.formatting import result_name
from leaderboard.src.display.utils import (
    DEFAULT_VIEW,
    TRACK_LABELS,
    TRACKS,
)
from leaderboard.src.explore import coilsets, model_graph
from leaderboard.src.leaderboard.read_evals import track_summary

logger = logging.getLogger(__name__)

DEFAULT_TRACK, DEFAULT_STRATUM = DEFAULT_VIEW


def _ranked(records: list[dict]) -> list[dict]:
    """Results in the leaderboard's default order: strict feasibility, then mean,
    in the default view (`DEFAULT_VIEW`)."""

    def key(record):
        summary = (
            track_summary(record["tracks"].get(DEFAULT_TRACK, {}), DEFAULT_STRATUM)
            or {}
        )
        return (
            summary.get("strict_feasibility", -1),
            summary.get("mean", -1),
        )

    return sorted(records, key=key, reverse=True)


def _result_choices(records: list[dict]) -> list[tuple[str, str]]:
    """The pickers' results, best first: (full name, scores parquet path in the
    results repo)."""
    return [
        (result_name(r["model"], r.get("onnx_path")), r["raw_scores_file"])
        for r in _ranked(records)
    ]


def render(get_records: Callable[[], list[dict]]) -> None:
    """Lay out the Explore tab's sub-tabs in the current Blocks context.

    `get_records` returns the current result records.
    """
    track_choices = [(TRACK_LABELS[t], t) for t in TRACKS]

    with gr.Tabs():
        with gr.Tab("Coilsets", id="coilsets") as coilsets_tab:
            with gr.Row(equal_height=True):
                coil_result = gr.Dropdown([], value=None, label="Result", scale=3)
                coil_track = gr.Radio(
                    track_choices, value=DEFAULT_TRACK, label="Track", scale=1
                )
            coil_instance = gr.Dropdown(
                [],
                value=None,
                label="Instance",
                info="Best score first. Type to search by boundary id.",
            )
            with gr.Row():
                with gr.Column(scale=3):
                    coil_plot = gr.Plot(
                        label="Coilset", show_label=False, elem_id="coilset-plot"
                    )
                with gr.Column(scale=2):
                    coil_summary = gr.Markdown(
                        label="Summary", elem_classes="markdown-text"
                    )

        with gr.Tab("Model graph", id="model-graph") as graph_tab:
            graph_result = gr.Dropdown([], value=None, label="Result")
            graph_view = gr.HTML(label="Model graph")

    # Coilsets.
    @functools.lru_cache(maxsize=8)
    def instance_frame(result, track):
        return coilsets.instances(coilsets.parquet_path(result), track)

    def list_instances(result, track):
        if not result:
            return gr.Dropdown(choices=[], value=None)
        df = instance_frame(result, track)
        # The index is each instance's row number in the parquet file.
        choices = [
            (coilsets.instance_label(cast(int, i), row), cast(int, i))
            for i, row in df.iterrows()
        ]
        return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

    def pick_result_once(current):
        """On a sub-tab's first visit, offer the current results and pick the best.

        Picking it fires the picker's change event, which loads that result. Later
        visits keep the user's choice.
        """
        if current is not None:
            return gr.skip()
        choices = _result_choices(get_records())
        return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

    def draw_coilset(result, track, instance):
        if not result or instance is None:
            return None, "This result has no instances in the selected track."
        row = instance_frame(result, track).loc[instance]
        summary = coilsets.summary_markdown(row)
        coilset = coilsets.load_coilset(coilsets.parquet_path(result), instance)
        if coilset is None:
            return None, summary + "\n\nThe model produced no coilset here."
        return coilsets.figure(coilset), summary

    coilsets_tab.select(
        pick_result_once, coil_result, coil_result, api_visibility="private"
    )
    gr.on(
        [coil_result.change, coil_track.change],
        list_instances,
        [coil_result, coil_track],
        coil_instance,
        api_name="coilset_instances",
    )
    coil_instance.change(
        draw_coilset,
        [coil_result, coil_track, coil_instance],
        [coil_plot, coil_summary],
        api_name="coilset",
    )

    # Model graph.
    def show_graph(result):
        if not result:
            return ""
        return model_graph.viewer_html(result)

    graph_tab.select(
        pick_result_once, graph_result, graph_result, api_visibility="private"
    )
    graph_result.change(show_graph, graph_result, graph_view, api_name="model_graph")
