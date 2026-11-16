"""The front page's layout and events: a Pareto plot over a Pareto-ranked table.

One set of controls (score, cost, track, stratum and a cost budget) drives both
the plot and the table, so they always show the same view. Search and the column
picker only change the table.

The results come from a function, called on every event, so the page follows
data that is refreshed while the app runs. Every control drives one event,
published as the `/pareto` API endpoint. Switching the cost axis also clears the
budget, whose unit it changes; that event is UI-only.
"""

from collections.abc import Callable

import gradio as gr
import pandas as pd

from leaderboard.src.display.leaderboard import search_rows, style_frame, write_csv
from leaderboard.src.display.utils import (
    COLS,
    DEFAULT_VIEW,
    INTEGER_COLS,
    METRIC_COLS,
    STRATUM_VIEWS,
    TRACK_LABELS,
    TRACKS,
    AutoEvalColumn,
    fields,
)
from leaderboard.src.frontpage import pareto

DEFAULT_TRACK, DEFAULT_STRATUM = DEFAULT_VIEW
DEFAULT_SCORE = next(iter(pareto.Y_AXES))
DEFAULT_COST = next(iter(pareto.X_AXES))

# What the front column shows for a result without a cost.
UNRANKED = "unranked"

# Rows to tone down: unranked, or over the budget. `--pf-muted` is set per
# light/dark mode in the app's CSS.
MUTED_CSS = "color: var(--pf-muted);"
# Rows on front 1, in the same SemiBold as the best value per metric column.
FRONT_CSS = "font-weight: 600;"

# The table's columns: the front, then the Score ranking table's.
ALL_COLUMNS = [pareto.FRONT, *COLS]
TYPES = {pareto.FRONT: "str"} | {c.name: c.type for c in fields(AutoEvalColumn)}
HIDDEN = [c.name for c in fields(AutoEvalColumn) if c.hidden]
# Offered in the column picker; the front and the cost axis are always shown.
PICKABLE = [c for c in COLS if c not in HIDDEN]
DEFAULT_COLUMNS = [
    c.name for c in fields(AutoEvalColumn) if c.displayed_by_default and not c.hidden
]
SEARCH_COLUMNS = [AutoEvalColumn.model.name, AutoEvalColumn.user_name.name]


def budget_label(cost: str) -> str:
    return f"Budget ({pareto.X_AXES[cost][2]})"


def _budget(value) -> float | None:
    """The budget as a positive number, or `None` for none."""
    return float(value) if value is not None and value > 0 else None


def front_as_text(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with its front column as text, `UNRANKED` where there is none."""
    front = df[pareto.FRONT]
    return df.assign(**{pareto.FRONT: front.astype(str).where(front.notna(), UNRANKED)})


def styled_table(ranked: pd.DataFrame, columns: list[str], cost: str, budget):
    """The Styler for `ranked`'s `columns`: front 1 SemiBold, unranked rows and
    rows over the budget muted, unranked rows marked in the front column."""
    front = ranked[pareto.FRONT]
    cost_col = pareto.X_AXES[cost][0]
    muted = front.isna()
    if budget is not None:
        muted |= ranked[cost_col] > budget
    shown = front_as_text(ranked[columns])

    on_front = (front == 1).fillna(False).to_numpy(bool)
    muted = muted.to_numpy(bool)

    def row_css(_):
        css = pd.DataFrame("", index=shown.index, columns=shown.columns)
        css.loc[on_front] = FRONT_CSS
        css.loc[muted] = css.loc[muted] + MUTED_CSS
        return css

    return style_frame(shown, METRIC_COLS, INTEGER_COLS).apply(row_css, axis=None)


def render(
    get_records: Callable[[], list[dict]], reload_on: Callable | None = None
) -> None:
    """Lay out the front page in the current Blocks context.

    `get_records` returns the current result records. `reload_on` (optional, an
    event such as `demo.load`) redraws the page from them.
    """

    # The parameter names are the API endpoint's parameter names.
    def view(score, cost, track, stratum, budget, search, columns):
        budget = _budget(budget)
        ranked = pareto.rank(
            pareto.view_frame(get_records(), track, stratum), score, cost
        )
        # Rounded only for display, after ranking.
        ranked = ranked.round(decimals=3)
        fig = (
            pareto.figure(ranked, score, cost, budget)
            if ranked[pareto.FRONT].notna().any()
            else None
        )
        note = pareto.note_markdown(ranked, score, cost, budget)
        # The front and the cost axis's column are always shown, in canonical order.
        cost_col = pareto.X_AXES[cost][0]
        keep = [
            c
            for c in ALL_COLUMNS
            if c == pareto.FRONT or c == cost_col or c in (columns or [])
        ]
        table_df = search_rows(ranked, search, SEARCH_COLUMNS)
        name = f"coilstellaration-pareto-{track}-{stratum}-{score}-{cost}".replace(
            " ", "-"
        )
        export = front_as_text(table_df[keep])
        return (
            fig,
            note,
            gr.update(
                value=styled_table(table_df, keep, cost, budget),
                datatype=[TYPES[c] for c in keep],
            ),
            write_csv(export, name),
        )

    def switch_cost(score, cost, track, stratum, search, columns):
        """Clear the budget, whose unit changed, and redraw without it."""
        return gr.Number(value=None, label=budget_label(cost)), *view(
            score, cost, track, stratum, None, search, columns
        )

    track_choices = [(TRACK_LABELS[t], t) for t in TRACKS]
    with gr.Row(equal_height=True):
        score_radio = gr.Radio(
            list(pareto.Y_AXES), value=DEFAULT_SCORE, label="Score", scale=1
        )
        cost_radio = gr.Radio(
            list(pareto.X_AXES), value=DEFAULT_COST, label="Cost", scale=2
        )
        track_radio = gr.Radio(
            track_choices, value=DEFAULT_TRACK, label="Track", scale=1
        )
        stratum_radio = gr.Radio(
            STRATUM_VIEWS, value=DEFAULT_STRATUM, label="Stratum", scale=2
        )
        budget_number = gr.Number(
            value=None,
            label=budget_label(DEFAULT_COST),
            info="Blank for none.",
            minimum=0,
            scale=1,
        )

    initial = view(
        DEFAULT_SCORE,
        DEFAULT_COST,
        DEFAULT_TRACK,
        DEFAULT_STRATUM,
        None,
        "",
        DEFAULT_COLUMNS,
    )
    # Labels name the API outputs; the plot's and note's aren't shown.
    plot = gr.Plot(
        value=initial[0], label="Pareto front", show_label=False, elem_id="pareto-plot"
    )
    note = gr.Markdown(value=initial[1], label="Notes", elem_classes="markdown-text")

    with gr.Row(equal_height=True):
        search_box = gr.Textbox(
            value="", label="Search", placeholder="Model or user", scale=2
        )
        column_picker = gr.Dropdown(
            choices=PICKABLE,
            value=DEFAULT_COLUMNS,
            multiselect=True,
            label="Columns",
            scale=3,
        )
    table = gr.Dataframe(
        value=initial[2]["value"],
        datatype=initial[2]["datatype"],
        interactive=False,
        wrap=True,
        show_row_numbers=True,
        label="Pareto ranking",
        show_label=False,
        elem_id="pareto-table",
    )
    download = gr.DownloadButton(
        "Download CSV", value=initial[3], size="sm", variant="secondary"
    )

    inputs = [
        score_radio,
        cost_radio,
        track_radio,
        stratum_radio,
        budget_number,
        search_box,
        column_picker,
    ]
    outputs = [plot, note, table, download]
    # The budget redraws as it's typed. Its change event isn't used, because
    # switching the cost axis clears it, and that event redraws already.
    gr.on(
        [
            score_radio.change,
            track_radio.change,
            stratum_radio.change,
            budget_number.input,
            search_box.change,
            column_picker.change,
        ],
        view,
        inputs=inputs,
        outputs=outputs,
        api_name="pareto",
        trigger_mode="always_last",
    )
    cost_radio.change(
        switch_cost,
        [
            score_radio,
            cost_radio,
            track_radio,
            stratum_radio,
            search_box,
            column_picker,
        ],
        [budget_number, *outputs],
        api_visibility="private",
    )
    if reload_on is not None:
        reload_on(view, inputs=inputs, outputs=outputs, api_visibility="private")
