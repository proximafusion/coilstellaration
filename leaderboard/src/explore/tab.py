"""The Explore tab's layout and events.

Nothing here runs until someone opens the tab or one of its sub-tabs: the Pareto
front is drawn when the Explore tab is selected, a result's coilsets are
downloaded when the Coilsets sub-tab is first opened, and Netron is framed only
when the Model graph sub-tab is. Gradio also mounts a tab's components only when
the tab is first shown.
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


def render(records: list[dict]) -> ExploreTab:
    """Lay out the Explore tab's sub-tabs in the current Blocks context."""
    records = _ranked(records)
    # Results in the pickers: (full name, scores parquet path in the results repo).
    result_choices = [(pareto.result_name(r), r["raw_scores_file"]) for r in records]
    default_result = result_choices[0][1] if result_choices else None
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
            pareto_plot = gr.Plot(show_label=False, elem_id="pareto-plot")
            pareto_note = gr.Markdown(elem_classes="markdown-text")

        with gr.Tab("Coilsets", id="coilsets") as coilsets_tab:
            with gr.Row(equal_height=True):
                coil_result = gr.Dropdown(
                    result_choices, value=default_result, label="Result", scale=3
                )
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
                    coil_plot = gr.Plot(show_label=False, elem_id="coilset-plot")
                with gr.Column(scale=2):
                    coil_summary = gr.Markdown(elem_classes="markdown-text")

        with gr.Tab("Model graph", id="model-graph") as graph_tab:
            graph_result = gr.Dropdown(
                result_choices, value=default_result, label="Result"
            )
            graph_view = gr.HTML()

    # Pareto front.
    def draw_pareto(y, x, track, stratum):
        df, missing = pareto.points(records, track, stratum, y, x)
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
    for control in pareto_inputs:
        control.change(draw_pareto, pareto_inputs, pareto_outputs)

    # Coilsets.
    @functools.lru_cache(maxsize=8)
    def instance_frame(raw_scores_file, track):
        return coilsets.instances(coilsets.parquet_path(raw_scores_file), track)

    def list_instances(raw_scores_file, track):
        if not raw_scores_file:
            return gr.Dropdown(choices=[], value=None)
        df = instance_frame(raw_scores_file, track)
        # The index is each instance's row number in the parquet file.
        choices = [
            (coilsets.instance_label(cast(int, i), row), cast(int, i))
            for i, row in df.iterrows()
        ]
        return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

    def list_instances_once(raw_scores_file, track, instance):
        # Only the first visit loads; later visits keep the user's choice.
        if instance is not None:
            return gr.skip()
        return list_instances(raw_scores_file, track)

    def draw_coilset(raw_scores_file, track, instance):
        if not raw_scores_file or instance is None:
            return None, "This result has no instances in the selected track."
        row = instance_frame(raw_scores_file, track).loc[instance]
        summary = coilsets.summary_markdown(row)
        coilset = coilsets.load_coilset(
            coilsets.parquet_path(raw_scores_file), instance
        )
        if coilset is None:
            return None, summary + "\n\nThe model produced no coilset here."
        return coilsets.figure(coilset), summary

    coilsets_tab.select(
        list_instances_once, [coil_result, coil_track, coil_instance], coil_instance
    )
    coil_result.change(list_instances, [coil_result, coil_track], coil_instance)
    coil_track.change(list_instances, [coil_result, coil_track], coil_instance)
    coil_instance.change(
        draw_coilset,
        [coil_result, coil_track, coil_instance],
        [coil_plot, coil_summary],
    )

    # Model graph.
    def show_graph(raw_scores_file):
        if not raw_scores_file:
            return ""
        return model_graph.viewer_html(raw_scores_file)

    def show_graph_once(raw_scores_file, current):
        return gr.skip() if current else show_graph(raw_scores_file)

    graph_tab.select(show_graph_once, [graph_result, graph_view], graph_view)
    graph_result.change(show_graph, graph_result, graph_view)

    pareto_defaults = tuple(c.value for c in pareto_inputs)
    return ExploreTab(pareto_inputs, pareto_outputs, draw_pareto, pareto_defaults)
