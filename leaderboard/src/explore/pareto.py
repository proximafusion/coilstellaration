"""The Pareto front of a score against a cost, one point per result.

A result is on the front when no other result scores at least as well for at
most the same cost, and strictly better on one of the two.
"""

import math

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
from leaderboard.src.leaderboard.read_evals import track_summary

# Score axes: the radio label, the summary key and the axis title. Higher is better.
Y_AXES = {
    "mean": ("mean", "Mean score"),
    "strict": ("strict_feasibility", "Strict feasibility"),
}
# Cost axes: the radio label and the axis title. Lower is better.
X_AXES = {
    "inference time": "Mean inference time per instance (ms)",
    "params": "Parameters",
}

# The page's font (see `css_html_js._FONT`).
FONT_FAMILY = "Inter Tight, system-ui, sans-serif"

# Trace names, also shown in the legend.
FRONT = "Pareto front"
DOMINATED = "Dominated"


def result_name(record: dict) -> str:
    """A result's full name: the model repo and its ONNX file."""
    return f"{record['model']} · {record.get('onnx_path', 'model.onnx')}"


def _cost(record: dict, summary: dict, x: str) -> float:
    if x == "params":
        return record.get("n_parameters") or math.nan
    # Stored in seconds; shown in milliseconds.
    return (summary.get("inference_time") or {}).get("mean", math.nan) * 1e3


def points(
    records: list[dict], track: str, stratum: str, y: str, x: str
) -> tuple[pd.DataFrame, int]:
    """One row per result with a score and a positive cost in this view.

    Returns the frame (columns name, user, x, y, on_front) and how many
    results scored in this view were left out for lack of a cost, such as
    results submitted before parameter counts were recorded.
    """
    y_key = Y_AXES[y][0]
    rows, missing_cost = [], 0
    for record in records:
        summary = track_summary(record["tracks"].get(track, {}), stratum)
        if not summary:
            continue
        score = summary.get(y_key, math.nan)
        cost = _cost(record, summary, x)
        if math.isnan(score):
            continue
        # The cost axis is logarithmic, so a zero cost can't be placed either.
        if math.isnan(cost) or cost <= 0:
            missing_cost += 1
            continue
        rows.append(
            {
                "name": result_name(record),
                "user": record.get("submitter", ""),
                "x": float(cost),
                "y": float(score),
            }
        )
    df = pd.DataFrame(rows, columns=["name", "user", "x", "y"])
    df["on_front"] = pareto_mask(df["x"].to_numpy(), df["y"].to_numpy())
    return df, missing_cost


def pareto_mask(cost: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Which points are on the front for a cost to minimise and a score to maximise.

    Of points with equal cost and score, only the first is kept.
    """
    on_front = np.zeros(len(cost), dtype=bool)
    best = -np.inf
    # Cheapest first; at equal cost the best score first, so it shadows the rest.
    for i in np.lexsort((-score, cost)):
        if score[i] > best:
            on_front[i] = True
            best = score[i]
    return on_front


def figure(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """The scatter of every result, with the front as a staircase line.

    Colours are for light mode. The app's CSS swaps them in dark mode, since a
    Plotly figure can't follow the page's colour scheme (see `custom_css`).
    """
    fig = go.Figure()
    hover = (
        "<b>%{customdata[0]}</b><br>user: %{customdata[1]}<br>"
        f"{x}: %{{x:.3g}}<br>{y}: %{{y:.3f}}<extra></extra>"
    )
    # A 2px ring in the surface colour keeps overlapping markers apart.
    ring = {"width": 2, "color": WHITE}
    dominated = df[~df["on_front"]]
    fig.add_trace(
        go.Scatter(
            name=DOMINATED,
            x=dominated["x"],
            y=dominated["y"],
            mode="markers",
            marker={"size": 10, "color": STEEL_BLUE, "line": ring},
            customdata=dominated[["name", "user"]],
            hovertemplate=hover,
        )
    )
    front = df[df["on_front"]].sort_values("x")
    fig.add_trace(
        go.Scatter(
            name=FRONT,
            x=front["x"],
            y=front["y"],
            # A staircase: each front point's score holds until the next is reached.
            mode="lines+markers",
            line={"shape": "hv", "width": 2, "color": MID_NAVY},
            marker={"size": 10, "color": MID_NAVY, "line": ring},
            customdata=front[["name", "user"]],
            hovertemplate=hover,
        )
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
        xaxis={**axis, "type": "log", "title": X_AXES[x]},
        yaxis={**axis, "title": Y_AXES[y][1], "rangemode": "tozero"},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
        hoverlabel={"bgcolor": NAVY, "bordercolor": STEEL_BLUE, "font_color": WHITE},
    )
    return fig


def front_markdown(df: pd.DataFrame, x: str, y: str) -> str:
    """The front's results as a table, cheapest first.

    Names are too long to label the points without overlap, so the plot leaves
    them to the hover and this table.
    """
    front = df[df["on_front"]].sort_values("x")
    if front.empty:
        return ""
    lines = [
        "**On the Pareto front**",
        "",
        f"| Result | User | {x} | {y} |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['name']} | {r['user']} | {r['x']:,.3g} | {r['y']:.3f} |"
        for _, r in front.iterrows()
    ]
    return "\n".join(lines)
