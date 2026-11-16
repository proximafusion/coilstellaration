"""Pareto ranking of results by a score against a cost, and its plot.

A result dominates another when it scores at least as well for at most the same
cost, and strictly better on one of the two. Front 1 is the results that no other
result dominates. Front 2 is front 1 of the results that remain once front 1 is
removed, and so on. A result without a recorded cost isn't ranked.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from leaderboard.src.display.css_html_js import (
    MID_NAVY,
    NAVY,
    STEEL_BLUE,
    SURFACE_COOL,
    WHITE,
)
from leaderboard.src.display.formatting import result_name
from leaderboard.src.display.utils import COLS, AutoEvalColumn
from leaderboard.src.leaderboard.read_evals import result_to_row

# Score axes: the radio label, then its table column and axis title. Higher is
# better. The first is the default.
Y_AXES = {
    "strict": (AutoEvalColumn.strict.name, "Strict feasibility"),
    "mean": (AutoEvalColumn.mean.name, "Mean score"),
}
# Cost axes: the radio label, then its table column, axis title and unit. Lower
# is better. The first is the default.
X_AXES = {
    "inference time": (
        AutoEvalColumn.time_ms.name,
        "Mean inference time per instance (ms)",
        "ms",
    ),
    "params": (AutoEvalColumn.n_parameters.name, "Parameters", "params"),
}

# The table column holding each result's front, 1 being the best. Results
# without a cost have none.
FRONT = "front"

# Trace names, also shown in the legend. The dark-mode CSS recolours the traces
# by their position, so their order is fixed: dominated, front, best in budget.
FRONT_TRACE = "Pareto front"
DOMINATED_TRACE = "Dominated"
BEST_TRACE = "Best within budget"

# Columns that name a result.
MODEL, ONNX = AutoEvalColumn.model.name, AutoEvalColumn.onnx_file.name

# Opacity of the points over the budget.
OVER_BUDGET_OPACITY = 0.25

# Abbreviated in the plot's labels, as `ABBREVIATION`, and explained in a
# footnote under the plot (`ABBREVIATION_NOTE`). Matched in any case.
ABBREVIATED = re.compile("coilstellaration", re.IGNORECASE)
ABBREVIATION = "CS"
ABBREVIATION_NOTE = "\\* In the plot's labels, CS stands for coilstellaration."

# Characters in a front label, shared by its owner and repo (see `short_name`).
LABEL_MAX = 28


def view_frame(records: list[dict], track: str, stratum: str) -> pd.DataFrame:
    """One leaderboard row (`COLS`) per result with data in this view, unrounded.

    The rows are those of the Score ranking table (see `result_to_row`); they
    aren't rounded, so rounding can't create ties that change the fronts.
    """
    rows = [
        row
        for record in records
        if (row := result_to_row(record, track, stratum)) is not None
    ]
    return pd.DataFrame.from_records(rows, columns=COLS)


def front_ranks(cost: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Each point's front (1 is the best) for a cost to minimise and a score to
    maximise; 0 for a point without a positive cost or without a score.

    A zero cost is left out because the plot's cost axis is logarithmic. Equal
    points share a front.
    """
    cost = np.asarray(cost, dtype=float)
    score = np.asarray(score, dtype=float)
    ranks = np.zeros(len(cost), dtype=int)
    remaining = np.flatnonzero(np.isfinite(cost) & (cost > 0) & np.isfinite(score))
    front = 0
    while remaining.size:
        front += 1
        c, s = cost[remaining], score[remaining]
        # Element [i, j] is whether point j dominates point i.
        no_worse = (c[None, :] <= c[:, None]) & (s[None, :] >= s[:, None])
        better = (c[None, :] < c[:, None]) | (s[None, :] > s[:, None])
        dominated = (no_worse & better).any(axis=1)
        ranks[remaining[~dominated]] = front
        remaining = remaining[dominated]
    return ranks


def rank(df: pd.DataFrame, score: str, cost: str) -> pd.DataFrame:
    """`df` with a `FRONT` column first, in Pareto order.

    Rows go by front, and within a front by score (highest first), then cost
    (lowest first). Unranked rows (`FRONT` missing) come last, highest score
    first.
    """
    y_col, x_col = Y_AXES[score][0], X_AXES[cost][0]
    ranks = front_ranks(df[x_col].to_numpy(float), df[y_col].to_numpy(float))
    order = (
        df.assign(_front=np.where(ranks > 0, ranks, np.iinfo(int).max))
        .sort_values(
            ["_front", y_col, x_col],
            ascending=[True, False, True],
            na_position="last",
            kind="stable",
        )
        .index
    )
    ranked = df.copy()
    # Front 0 (no cost) is shown as missing.
    ranked.insert(0, FRONT, pd.Series(ranks, index=df.index, dtype="Int64"))
    ranked[FRONT] = ranked[FRONT].mask(ranked[FRONT] == 0)
    return ranked.loc[order]


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "..."


def short_name(model: str) -> str:
    """A result's label on the plot: its model repo id, `owner/repo`, cut to
    `LABEL_MAX` characters in total (not counting the "/" or "...").
    "coilstellaration" is shortened to `ABBREVIATION` first.

    Each part gets half, and a part shorter than its half leaves the rest to the
    other. A part that is cut short ends in "...". The hover shows the full name.
    """
    owner, _, repo = ABBREVIATED.sub(ABBREVIATION, model).rpartition("/")
    if not owner:
        return _cut(repo, LABEL_MAX)
    owner_len = min(len(owner), max(LABEL_MAX // 2, LABEL_MAX - len(repo)))
    return f"{_cut(owner, owner_len)}/{_cut(repo, LABEL_MAX - owner_len)}"


def best_within(df: pd.DataFrame, score: str, cost: str, budget: float | None):
    """The ranked row with the highest score at a cost within `budget` (the
    cheapest of equals), or `None` if no ranked row is within it."""
    if budget is None:
        return None
    y_col, x_col = Y_AXES[score][0], X_AXES[cost][0]
    within = df[df[FRONT].notna() & (df[x_col] <= budget)]
    if within.empty:
        return None
    return within.sort_values([y_col, x_col], ascending=[False, True]).iloc[0]


def figure(
    df: pd.DataFrame, score: str, cost: str, budget: float | None = None
) -> go.Figure:
    """The scatter of every ranked row of `df` (see `rank`), with front 1 as a
    labelled staircase.

    With a `budget`, a dashed line marks it, the points over it are faded and the
    best point within it is ringed.

    Colours are for light mode. The app's CSS swaps them in dark mode, since a
    Plotly figure can't follow the page's colour scheme (see `custom_css`).
    """
    y_col, y_title = Y_AXES[score]
    x_col, x_title, _ = X_AXES[cost]
    ranked = df[df[FRONT].notna()]
    names = list(zip(ranked[MODEL], ranked[ONNX]))
    points = pd.DataFrame(
        {
            "x": ranked[x_col].astype(float),
            "y": ranked[y_col].astype(float),
            "front": ranked[FRONT].astype(int),
            "name": [result_name(m, f) for m, f in names],
            "label": [short_name(m) for m, _ in names],
            "user": ranked[AutoEvalColumn.user_name.name],
        }
    )
    within = points["x"] <= budget if budget is not None else points["x"] == points["x"]
    points["opacity"] = np.where(within, 1.0, OVER_BUDGET_OPACITY)

    fig = go.Figure()
    hover = (
        "<b>%{customdata[0]}</b><br>user: %{customdata[1]}<br>"
        "front: %{customdata[2]}<br>"
        f"{cost}: %{{x:.3g}}<br>{score}: %{{y:.3f}}<extra></extra>"
    )
    custom = ["name", "user", "front"]
    # A 2px ring in the surface colour keeps overlapping markers apart.
    ring = {"width": 2, "color": WHITE}
    dominated = points[points["front"] > 1]
    fig.add_trace(
        go.Scatter(
            name=DOMINATED_TRACE,
            x=dominated["x"],
            y=dominated["y"],
            mode="markers",
            marker={
                "size": 10,
                "color": STEEL_BLUE,
                "line": ring,
                "opacity": dominated["opacity"],
            },
            customdata=dominated[custom],
            hovertemplate=hover,
        )
    )
    front = points[points["front"] == 1].sort_values("x")
    fig.add_trace(
        go.Scatter(
            name=FRONT_TRACE,
            x=front["x"],
            y=front["y"],
            # A staircase: each front point's score holds until the next is reached.
            mode="lines+markers+text",
            line={"shape": "hv", "width": 2, "color": MID_NAVY},
            marker={
                "size": 10,
                "color": MID_NAVY,
                "line": ring,
                "opacity": front["opacity"],
            },
            # Above and left of a front point is empty: anything there would
            # dominate it.
            # The cheapest point's label goes right, so it isn't cut off at the
            # plot's left edge.
            text=front["label"],
            textposition=["top right"] + ["top left"] * (len(front) - 1),
            textfont={
                "size": 12,
                "color": [NAVY if o == 1 else STEEL_BLUE for o in front["opacity"]],
            },
            customdata=front[custom],
            hovertemplate=hover,
        )
    )
    if budget is not None:
        best = points[within].sort_values(["y", "x"], ascending=[False, True]).head(1)
        fig.add_trace(
            go.Scatter(
                name=BEST_TRACE,
                x=best["x"],
                y=best["y"],
                mode="markers",
                marker={
                    "size": 22,
                    "symbol": "circle-open",
                    "color": NAVY,
                    "line": {"width": 2.5},
                },
                customdata=best[custom],
                hovertemplate=hover,
            )
        )
        # Named in the legend, not labelled on the plot: an annotation on a log
        # axis is placed by the log of its position, unlike the line itself,
        # and Plotly would stretch the axis to reach it.
        fig.add_vline(
            x=budget,
            line={"dash": "dash", "width": 1.5, "color": STEEL_BLUE},
            name=f"Budget ({budget:,.4g} {X_AXES[cost][2]})",
            showlegend=True,
        )
    axis = {
        "gridcolor": SURFACE_COOL,
        "linecolor": STEEL_BLUE,
        "zeroline": False,
        "ticks": "outside",
        "tickangle": 0,
        "tickcolor": STEEL_BLUE,
    }
    fig.update_layout(
        height=520,
        margin={"l": 64, "r": 24, "t": 48, "b": 56},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter Tight, system-ui, sans-serif",
            "size": 14,
            "color": NAVY,
        },
        xaxis={**axis, "type": "log", "title": x_title},
        yaxis={**axis, "title": y_title, "rangemode": "tozero"},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
        hoverlabel={"bgcolor": NAVY, "bordercolor": STEEL_BLUE, "font_color": WHITE},
    )
    return fig


def note_markdown(
    df: pd.DataFrame, score: str, cost: str, budget: float | None = None
) -> str:
    """The text under the plot: the best result within the budget, how many
    results aren't plotted for lack of a cost, and, when a label on the plot
    abbreviates "coilstellaration", a footnote saying so."""
    _, _, unit = X_AXES[cost]
    lines = []
    if budget is not None:
        best = best_within(df, score, cost, budget)
        if best is None:
            lines.append(f"No ranked result costs at most {budget:,.4g} {unit}.")
        else:
            x_col = X_AXES[cost][0]
            lines.append(
                f"Best within {budget:,.4g} {unit}: "
                f"**{result_name(best[MODEL], best[ONNX])}**, "
                f"{score} {best[Y_AXES[score][0]]:.3f} at {best[x_col]:,.3g} {unit}."
            )
    unranked = int(df[FRONT].isna().sum())
    if unranked:
        what = "parameter count" if cost == "params" else "inference time"
        results = f"result{'s' if unranked != 1 else ''}"
        lines.append(
            f"Not plotted: {unranked} {results} without a recorded {what}, listed "
            "as unranked at the bottom of the table."
        )
    # Only front 1 is labelled.
    labelled = df.loc[df[FRONT] == 1, MODEL]
    if labelled.str.contains(ABBREVIATED).any():
        lines.append(ABBREVIATION_NOTE)
    return "\n\n".join(lines)
