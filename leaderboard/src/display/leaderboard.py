"""A minimal local replacement for the `gradio_leaderboard` package.

Provides just the slice of its API that `app_leaderboard.py` uses: a
read-only table with column selection, free-text search and a checkbox
filter. Built entirely from stock gradio components.
"""

import re
import tempfile
from pathlib import Path
from typing import Any, cast

import gradio as gr
import pandas as pd


class ColumnFilter:
    """A checkbox-group filter over the distinct values of one column."""

    def __init__(self, column, type="checkboxgroup", label=None):
        self.column = column
        self.type = type
        self.label = label or column


class SelectColumns:
    """Which columns are shown, and which the user may not hide."""

    def __init__(
        self,
        default_selection=None,
        cant_deselect=None,
        label="Select columns:",
        info=None,
    ):
        self.default_selection = default_selection or []
        self.cant_deselect = cant_deselect or []
        self.label = label
        self.info = info


def search_rows(df: pd.DataFrame, query: str, columns) -> pd.DataFrame:
    """Rows where any of `columns` contains `query` (case-insensitive, literal)."""
    if not query:
        return df
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask |= (
            df[col].astype(str).str.contains(query, case=False, na=False, regex=False)
        )
    return df[mask]


# How the best value in each metric column is marked: SemiBold on a pale surface.
# This is the one place heavier than the design system's Medium, so the winners
# stand out from the Medium column headers. `--pf-best-bg` is set per light/dark
# mode in the app's CSS.
BEST_VALUE_CSS = "font-weight: 600; background: var(--pf-best-bg);"


def style_frame(df: pd.DataFrame, highlight_columns, integer_columns=()):
    """A Styler that marks each `highlight_columns` maximum and formats numbers.

    Gradio renders the Styler's display values, so every number is formatted
    here: `integer_columns` as whole numbers with thousands separators, the other
    numeric columns with 3 decimals. Missing values show as blank cells.
    """
    present = [c for c in highlight_columns if c in df.columns]
    integers = [c for c in integer_columns if c in df.columns]
    decimals = [
        c
        for c in df.columns
        if c not in integers and pd.api.types.is_numeric_dtype(df[c])
    ]
    return (
        df.style.highlight_max(subset=present, props=BEST_VALUE_CSS)
        .format(precision=3, na_rep="", subset=decimals)
        .format("{:,.0f}", na_rep="", subset=integers)
    )


_HTML_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(r'<a\b[^>]*\bhref="([^"]+)"[^>]*>\s*link\s*</a>')


def _plain(value):
    """A cell as plain text: a "link" anchor becomes its URL, other HTML its text."""
    if not isinstance(value, str):
        return value
    if match := _LINK.fullmatch(value.strip()):
        return match.group(1)
    return _HTML_TAG.sub("", value)


def write_csv(df: pd.DataFrame, name: str) -> str:
    """Write `df` to a new CSV file called `<name>.csv`; return its path.

    "link" cells become their URL, and other HTML links are reduced to their
    text.
    Each call gets its own directory, so concurrent users don't overwrite each
    other's files.
    """
    plain = df.copy()
    for col in plain.columns:
        if plain[col].dtype == object:
            plain[col] = plain[col].map(_plain)
    path = Path(tempfile.mkdtemp(prefix="leaderboard-csv-")) / f"{name}.csv"
    plain.to_csv(path, index=False)
    return str(path)


def Leaderboard(
    value,
    datatype=None,
    select_columns=None,
    search_columns=None,
    hide_columns=None,
    filter_columns=None,
    interactive=False,
    frames=None,
    tracks=None,
    strata=None,
    default_view=None,
    track_labels=None,
    track_label="Track",
    stratum_label="Stratum",
    stratum_info=None,
    highlight_columns=None,
    integer_columns=None,
    label=None,
    download_name="leaderboard",
    api_name="leaderboard",
):
    """Render the leaderboard controls and table, returning the table component.

    `value.columns` is the full, ordered set of columns; `datatype` is the
    parallel list of gradio cell types. Selecting/searching/filtering all
    re-derive the displayed frame from the active base frame, never mutating it.

    `highlight_columns` (optional) are the metric columns whose best (maximum)
    value is highlighted. `integer_columns` (optional) are shown as whole numbers.
    `label` is the table's label. A download button offers the displayed table as
    `<download_name>-<track>-<stratum>.csv`.

    Every control drives one event, published as the `api_name` endpoint: it takes
    the columns, search text, track label and stratum, and returns the table and
    the CSV.

    `frames` (optional) maps a `(track, stratum)` key to a same-columns frame;
    with `tracks`/`strata` giving the axis orders, a Track radio and a Stratum
    radio let the user pick which frame the table shows. `track_labels` maps a
    track key to the label shown on the radio. `default_view` (optional) is the
    `(track, stratum)` selected on load; it defaults to the first of each.
    `value` is the default frame.
    """
    frames = dict(frames) if frames else {(None, None): value}
    tracks = list(tracks) if tracks else [None]
    strata = list(strata) if strata else [None]
    track_labels = track_labels or {t: str(t) for t in tracks}
    default_track, default_stratum = default_view or (tracks[0], strata[0])

    all_columns = list(value.columns)
    type_of: dict[str, Any] = dict(
        zip(all_columns, datatype or ["str"] * len(all_columns))
    )

    select_columns = select_columns or SelectColumns(default_selection=all_columns)
    search_columns = search_columns or []
    hide_columns = hide_columns or []
    filter_columns = filter_columns or []
    highlight_columns = highlight_columns or []
    integer_columns = integer_columns or []

    # Radios carry the human labels; map back to the (track, stratum) frame key.
    label_to_track = {label: t for t, label in track_labels.items()}

    # Columns the user can choose from (drop the always-hidden ones).
    selectable = [c for c in all_columns if c not in hide_columns]

    def csv_name(track, stratum):
        return "-".join(str(p) for p in (download_name, track, stratum) if p)

    # The parameter names are the API endpoint's parameter names.
    def view(columns, search, track, stratum, *filter_values):
        track = label_to_track.get(track, track)
        df = frames.get((track, stratum), value)
        df = search_rows(df, search, search_columns)
        for flt, chosen in zip(filter_columns, filter_values):
            if chosen and flt.column in df.columns:
                df = df[df[flt.column].astype(str).isin(chosen)]
        # Honour the selection but keep the canonical column order, and never
        # drop columns the user isn't allowed to deselect.
        keep = [
            c for c in all_columns if c in columns or c in select_columns.cant_deselect
        ]
        df = df[keep]
        return (
            gr.update(
                value=style_frame(df, highlight_columns, integer_columns),
                datatype=[type_of[c] for c in keep],
            ),
            write_csv(df, csv_name(track, stratum)),
        )

    # Search and the track/stratum switches share a row above the table; the column
    # picker is folded away below them. A plain State stands in for an axis with a
    # single value so `view`'s signature stays uniform.
    with gr.Row(equal_height=True):
        search_box = (
            gr.Textbox(
                value="",
                label="Search",
                placeholder="Model or user",
                scale=2,
            )
            if search_columns
            else None
        )
        track_radio = (
            gr.Radio(
                choices=[track_labels[t] for t in tracks],
                value=track_labels[default_track],
                label=track_label,
                scale=1,
            )
            if len(tracks) > 1
            else None
        )
        stratum_radio = (
            gr.Radio(
                # Only built for real strata, so never the `[None]` placeholder.
                choices=cast(list[str], strata),
                value=default_stratum,
                label=stratum_label,
                info=stratum_info,
                scale=2,
            )
            if len(strata) > 1
            else None
        )
    # Columns that can't be hidden aren't offered: they are always shown.
    column_picker = gr.Dropdown(
        choices=[c for c in selectable if c not in select_columns.cant_deselect],
        value=[
            c
            for c in select_columns.default_selection
            if c not in select_columns.cant_deselect
        ],
        multiselect=True,
        label=select_columns.label,
        info=select_columns.info,
    )
    filter_boxes = [
        gr.CheckboxGroup(
            choices=sorted(value[flt.column].astype(str).unique())
            if not value.empty
            else [],
            label=flt.label,
        )
        for flt in filter_columns
    ]

    default_frame = frames[(default_track, default_stratum)]
    default_keep = [
        c
        for c in all_columns
        if c in select_columns.default_selection or c in select_columns.cant_deselect
    ]
    table = gr.Dataframe(
        value=style_frame(
            default_frame[default_keep], highlight_columns, integer_columns
        ),
        datatype=[type_of[c] for c in default_keep],
        interactive=interactive,
        wrap=True,
        show_row_numbers=True,
        label=label,
        show_label=False,
        elem_id="leaderboard-table",
    )
    # Its file always matches the table: every view change rewrites it.
    download = gr.DownloadButton(
        "Download CSV",
        value=write_csv(
            default_frame[default_keep], csv_name(default_track, default_stratum)
        ),
        size="sm",
        variant="secondary",
    )

    controls = (
        [column_picker]
        + ([track_radio] if track_radio else [])
        + ([stratum_radio] if stratum_radio else [])
        + ([search_box] if search_box else [])
        + filter_boxes
    )
    inputs = [
        column_picker,
        search_box or gr.State(""),
        track_radio or gr.State(track_labels[default_track]),
        stratum_radio or gr.State(default_stratum),
    ] + filter_boxes
    gr.on(
        [control.change for control in controls],
        view,
        inputs=inputs,
        outputs=[table, download],
        api_name=api_name,
    )

    return table
