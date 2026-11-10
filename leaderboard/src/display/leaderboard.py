"""A minimal local replacement for the `gradio_leaderboard` package.

Provides just the slice of its API that `app_leaderboard.py` uses: a
read-only table with column selection, free-text search and a checkbox
filter. Built entirely from stock gradio components.
"""

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
        self, default_selection=None, cant_deselect=None, label="Select columns:"
    ):
        self.default_selection = default_selection or []
        self.cant_deselect = cant_deselect or []
        self.label = label


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
    track_labels=None,
    track_label="Track",
    stratum_label="Stratum",
):
    """Render the leaderboard controls and table, returning the table component.

    `value.columns` is the full, ordered set of columns; `datatype` is the
    parallel list of gradio cell types. Selecting/searching/filtering all
    re-derive the displayed frame from the active base frame, never mutating it.

    `frames` (optional) maps a `(track, stratum)` key to a same-columns frame;
    with `tracks`/`strata` giving the axis orders, a Track radio and a Stratum
    radio let the user pick which frame the table shows. `track_labels` maps a
    track key to the label shown on the radio. `value` is the default frame.
    """
    frames = dict(frames) if frames else {(None, None): value}
    tracks = list(tracks) if tracks else [None]
    strata = list(strata) if strata else [None]
    track_labels = track_labels or {t: str(t) for t in tracks}
    default_track, default_stratum = tracks[0], strata[0]

    all_columns = list(value.columns)
    type_of = dict(zip(all_columns, datatype or ["str"] * len(all_columns)))

    select_columns = select_columns or SelectColumns(default_selection=all_columns)
    search_columns = search_columns or []
    hide_columns = hide_columns or []
    filter_columns = filter_columns or []

    # Radios carry the human labels; map back to the (track, stratum) frame key.
    label_to_track = {label: t for t, label in track_labels.items()}

    # Columns the user can choose from (drop the always-hidden ones).
    selectable = [c for c in all_columns if c not in hide_columns]

    def view(selected, query, track_choice, stratum, *filter_values):
        track = label_to_track.get(track_choice, track_choice)
        df = frames.get((track, stratum), value)
        df = search_rows(df, query, search_columns)
        for flt, chosen in zip(filter_columns, filter_values):
            if chosen and flt.column in df.columns:
                df = df[df[flt.column].astype(str).isin(chosen)]
        # Honour the selection but keep the canonical column order, and never
        # drop columns the user isn't allowed to deselect.
        keep = [
            c for c in all_columns if c in selected or c in select_columns.cant_deselect
        ]
        df = df[keep]
        return gr.update(value=df, datatype=[type_of[c] for c in keep])

    # The track/stratum switches sit above the table; a plain State stands in
    # for an axis with a single value so `view`'s signature stays uniform.
    track_radio = (
        gr.Radio(
            choices=[track_labels[t] for t in tracks],
            value=track_labels[default_track],
            label=track_label,
        )
        if len(tracks) > 1
        else None
    )
    stratum_radio = (
        gr.Radio(choices=strata, value=default_stratum, label=stratum_label)
        if len(strata) > 1
        else None
    )
    column_picker = gr.CheckboxGroup(
        choices=selectable,
        value=select_columns.default_selection,
        label=select_columns.label,
    )
    search_box = (
        gr.Textbox(label="Search", placeholder="Search…") if search_columns else None
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
        value=default_frame[default_keep],
        datatype=[type_of[c] for c in default_keep],
        interactive=interactive,
        wrap=True,
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
    for control in controls:
        control.change(view, inputs=inputs, outputs=table)

    return table
