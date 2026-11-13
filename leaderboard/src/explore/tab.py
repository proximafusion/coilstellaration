"""The Explore tab's layout and events.

Nothing here runs until someone opens the tab or one of its sub-tabs: the Pareto
front is drawn when the Explore tab is selected, a result's coilsets are
downloaded when the Coilsets sub-tab is first opened, and Netron is framed only
when the Model graph sub-tab is. Gradio also mounts a tab's components only when
the tab is first shown.

The results come from a function, called on every event, so the tab follows data
that is refreshed while the app runs. The Coilsets and Model graph pickers are
filled with the current results when their sub-tab is first opened.

The events that load a sub-tab on first visit are UI-only (`api_visibility=
"private"`). The rest are published as API endpoints: `/pareto`,
`/coilset_instances`, `/coilset` and `/model_graph`.
"""

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import gradio as gr

from leaderboard.src.display.utils import (
    DEFAULT_VIEW,
    STRATUM_VIEWS,
    TRACK_LABELS,
    TRACKS,
)
from leaderboard.src.explore import coilsets, model_graph, pareto
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


@dataclass
class ExploreTab:
    """The components and handler that the app wires to the Explore tab itself."""

    pareto_inputs: list
    pareto_outputs: list
    draw_pareto: Callable
    # The Pareto controls' values on a freshly loaded page.
    pareto_defaults: tuple


def _result_choices(records: list[dict]) -> list[tuple[str, str]]:
    """The pickers' results, best first: (full name, scores parquet path in the
    results repo)."""
    return [(pareto.result_name(r), r["raw_scores_file"]) for r in _ranked(records)]


def render(get_records: Callable[[], list[dict]]) -> ExploreTab:
    """Lay out the Explore tab's sub-tabs in the current Blocks context.

    `get_records` returns the current result records.
    """
    track_choices = [(TRACK_LABELS[t], t) for t in TRACKS]

    with gr.Tabs():
        with gr.Tab("Pareto front", id="pareto"):
            with gr.Row(equal_height=True):
                score_radio = gr.Radio(list(pareto.Y_AXES), value="mean", label="Score")
                cost_radio = gr.Radio(
                    list(pareto.X_AXES), value="inference time", label="Cost"
                )
                pareto_track = gr.Radio(
                    track_choices, value=DEFAULT_TRACK, label="Track"
                )
                pareto_stratum = gr.Radio(
                    STRATUM_VIEWS, value=DEFAULT_STRATUM, label="Stratum", scale=2
                )
            # Labels name the API outputs; they aren't shown.
            pareto_plot = gr.Plot(
                label="Pareto front", show_label=False, elem_id="pareto-plot"
            )
            pareto_note = gr.Markdown(label="Notes", elem_classes="markdown-text")

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

    # Pareto front.
    # Handler parameter names are the API endpoints' parameter names.
    def draw_pareto(score, cost, track, stratum):
        y, x = score, cost
        df, missing = pareto.points(get_records(), track, stratum, y, x)
        note = ""
        if missing:
            what = "parameter count" if x == "params" else "inference time"
            note = (
                f"Not shown: {missing} result{'s' if missing != 1 else ''} "
                f"without a recorded {what}."
            )
        if df.empty:
            return None, note
        note = "\n\n".join(filter(None, [pareto.front_markdown(df, x, y), note]))
        return pareto.figure(df, x, y), note

    pareto_inputs = [score_radio, cost_radio, pareto_track, pareto_stratum]
    pareto_outputs = [pareto_plot, pareto_note]
    gr.on(
        [control.change for control in pareto_inputs],
        draw_pareto,
        pareto_inputs,
        pareto_outputs,
        api_name="pareto",
    )

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

    pareto_defaults = tuple(c.value for c in pareto_inputs)
    return ExploreTab(pareto_inputs, pareto_outputs, draw_pareto, pareto_defaults)
