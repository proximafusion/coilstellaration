"""The Proxima Fusion look for both apps, from the Proxima Fusion design system.

Tokens follow the design system's `colors_and_type.css`: navy `#00102E` carries
the page, teal `#34CEB7` is a rare accent (focus rings, the selected tab, rules
under section headers, bullets), surfaces are white or pale blue-grey `#E1EAF2`,
and type is Inter Tight in two weights only (400 body, 500 headings and
emphasis). Text colour is binary: navy on light, white on navy, with no grey
tier. Corners are tight (4 px) and the UI is flat (no shadows).
"""

from pathlib import Path

import gradio as gr

# Design-system tokens.
NAVY = "#00102E"
TEAL = "#34CEB7"
WHITE = "#FFFFFF"
SURFACE_PALE = "#E1EAF2"
SURFACE_COOL = "#CEDDE8"
STEEL_BLUE = "#6293A0"
MID_NAVY = "#143F5F"
ALERT_RED = "#CC0000"

TABLE_ZEBRA_SCREEN = "#EEF3F7"

# Dark mode: navy page with lifted navy cards and white text. Edges are
# steel-blue, as in light mode: 5.6:1 on the navy page, 4.6:1 on cards.
NAVY_CARD = "#0D2447"
NAVY_TINT = "#0A1A36"
NAVY_TINT_2 = "#14233F"
BORDER_ON_DARK = STEEL_BLUE
CONTROL_BORDER_ON_DARK = STEEL_BLUE

# Contrast (WCAG 2.2 asks 3:1 for the edges of controls, 4.5:1 for text). Pale
# surfaces can't reach 3:1 against white or each other (white on #E1EAF2 is
# 1.2:1), so in light mode edges carry the contrast: the page is white, and cards,
# inputs, radio chips and checkboxes are edged in steel-blue (3.4:1 on white).
# Secondary buttons have a navy outline. Teal is 2:1 on white, too faint to mark
# state, so the selected tab and focus ring are navy; on navy, teal is 9.6:1 and
# is kept.

# Load every weight in use: 400 (body), 500 (Medium: headings, labels, emphasis)
# and 600 (the best value in each table column). Gradio loads only 400 and 600 by
# default, which renders every Medium as 400.
_FONT = (
    gr.themes.GoogleFont("Inter Tight", weights=(400, 500, 600)),
    "system-ui",
    "sans-serif",
)

theme = gr.themes.Base(
    font=_FONT,
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    # Page and blocks: white, flat, hairline borders.
    body_background_fill=WHITE,
    body_background_fill_dark=NAVY,
    body_text_color=NAVY,
    body_text_color_dark=WHITE,
    body_text_color_subdued=NAVY,
    body_text_color_subdued_dark=WHITE,
    body_text_weight="400",
    background_fill_primary=WHITE,
    background_fill_primary_dark=NAVY,
    background_fill_secondary=SURFACE_PALE,
    background_fill_secondary_dark=NAVY_TINT,
    block_background_fill=WHITE,
    block_background_fill_dark=NAVY_CARD,
    block_border_color=STEEL_BLUE,
    block_border_color_dark=BORDER_ON_DARK,
    block_shadow="none",
    block_shadow_dark="none",
    block_radius="4px",
    panel_background_fill=WHITE,
    panel_background_fill_dark=NAVY_CARD,
    panel_border_color=STEEL_BLUE,
    panel_border_color_dark=BORDER_ON_DARK,
    border_color_primary=STEEL_BLUE,
    border_color_primary_dark=BORDER_ON_DARK,
    border_color_accent=NAVY,
    border_color_accent_dark=TEAL,
    border_color_accent_subdued=STEEL_BLUE,
    border_color_accent_subdued_dark=TEAL,
    color_accent=NAVY,
    color_accent_soft=SURFACE_PALE,
    color_accent_soft_dark=NAVY_TINT_2,
    # Labels and titles: navy (white on dark), Medium weight.
    block_label_background_fill=WHITE,
    block_label_background_fill_dark=NAVY_CARD,
    block_label_text_color=NAVY,
    block_label_text_color_dark=WHITE,
    block_label_text_weight="500",
    block_label_border_color=STEEL_BLUE,
    block_label_border_color_dark=BORDER_ON_DARK,
    block_label_shadow="none",
    block_title_text_color=NAVY,
    block_title_text_color_dark=WHITE,
    block_title_text_weight="500",
    block_info_text_color=NAVY,
    block_info_text_color_dark=WHITE,
    accordion_text_color=NAVY,
    accordion_text_color_dark=WHITE,
    # Links: navy, never recoloured on hover.
    link_text_color=NAVY,
    link_text_color_dark=WHITE,
    link_text_color_hover=NAVY,
    link_text_color_hover_dark=WHITE,
    link_text_color_active=NAVY,
    link_text_color_active_dark=WHITE,
    link_text_color_visited=NAVY,
    link_text_color_visited_dark=WHITE,
    # Inputs: white with a hairline border; teal border on focus.
    input_background_fill=WHITE,
    input_background_fill_dark=NAVY,
    input_background_fill_focus=WHITE,
    input_background_fill_focus_dark=NAVY,
    input_border_color=STEEL_BLUE,
    input_border_color_dark=CONTROL_BORDER_ON_DARK,
    input_border_color_hover=NAVY,
    input_border_color_hover_dark=WHITE,
    input_border_color_focus=NAVY,
    input_border_color_focus_dark=TEAL,
    input_placeholder_color=STEEL_BLUE,
    input_placeholder_color_dark=SURFACE_COOL,
    input_shadow="none",
    input_shadow_dark="none",
    input_shadow_focus="none",
    input_shadow_focus_dark="none",
    input_radius="4px",
    # Gradio's base theme draws no edge on inputs (0 px), and checkboxes and
    # radio chips inherit it, so without this the edge colours never show.
    input_border_width="1px",
    # Radios and checkboxes: white chips edged in steel-blue; the selected one
    # is navy (white on dark).
    checkbox_label_background_fill=WHITE,
    checkbox_label_background_fill_dark=NAVY,
    checkbox_label_background_fill_hover=SURFACE_PALE,
    checkbox_label_background_fill_hover_dark=NAVY_TINT_2,
    checkbox_label_background_fill_selected=NAVY,
    checkbox_label_background_fill_selected_dark=WHITE,
    checkbox_label_text_color=NAVY,
    checkbox_label_text_color_dark=WHITE,
    checkbox_label_text_color_selected=WHITE,
    checkbox_label_text_color_selected_dark=NAVY,
    checkbox_label_border_color=STEEL_BLUE,
    checkbox_label_border_color_dark=CONTROL_BORDER_ON_DARK,
    checkbox_label_border_color_hover=NAVY,
    checkbox_label_border_color_hover_dark=WHITE,
    checkbox_label_border_color_selected=NAVY,
    checkbox_label_border_color_selected_dark=WHITE,
    checkbox_label_shadow="none",
    checkbox_label_text_weight="400",
    checkbox_background_color=WHITE,
    checkbox_background_color_dark=NAVY,
    checkbox_background_color_selected=NAVY,
    checkbox_background_color_selected_dark=TEAL,
    checkbox_border_color=STEEL_BLUE,
    checkbox_border_color_dark=CONTROL_BORDER_ON_DARK,
    checkbox_border_color_selected=NAVY,
    checkbox_border_color_selected_dark=TEAL,
    checkbox_border_color_focus=NAVY,
    checkbox_border_color_hover=NAVY,
    checkbox_border_color_hover_dark=WHITE,
    checkbox_border_color_focus_dark=TEAL,
    checkbox_border_radius="2px",
    checkbox_shadow="none",
    # Buttons: navy primary, pale secondary; flat, 4 px corners, Medium weight.
    button_primary_background_fill=NAVY,
    button_primary_background_fill_dark=WHITE,
    button_primary_background_fill_hover=MID_NAVY,
    button_primary_background_fill_hover_dark=SURFACE_PALE,
    button_primary_text_color=WHITE,
    button_primary_text_color_dark=NAVY,
    button_primary_text_color_hover=WHITE,
    button_primary_text_color_hover_dark=NAVY,
    button_primary_border_color=NAVY,
    button_primary_border_color_dark=WHITE,
    button_primary_border_color_hover=MID_NAVY,
    button_primary_border_color_hover_dark=SURFACE_PALE,
    button_primary_shadow="none",
    button_primary_shadow_hover="none",
    button_primary_shadow_active="none",
    button_secondary_background_fill=WHITE,
    button_secondary_background_fill_dark=NAVY_CARD,
    button_secondary_background_fill_hover=SURFACE_PALE,
    button_secondary_background_fill_hover_dark=NAVY_TINT_2,
    button_secondary_text_color=NAVY,
    button_secondary_text_color_dark=WHITE,
    button_secondary_text_color_hover=NAVY,
    button_secondary_text_color_hover_dark=WHITE,
    button_secondary_border_color=NAVY,
    button_secondary_border_color_dark=WHITE,
    button_secondary_border_color_hover=NAVY,
    button_secondary_border_color_hover_dark=WHITE,
    button_secondary_shadow="none",
    button_secondary_shadow_hover="none",
    button_secondary_shadow_active="none",
    button_large_radius="4px",
    button_medium_radius="4px",
    button_small_radius="4px",
    button_large_text_weight="500",
    button_medium_text_weight="500",
    button_small_text_weight="500",
    # Tables: navy text, white and zebra rows, hairline rules.
    table_text_color=NAVY,
    table_text_color_dark=WHITE,
    table_odd_background_fill=WHITE,
    table_odd_background_fill_dark=NAVY_CARD,
    table_even_background_fill=SURFACE_PALE,
    table_even_background_fill_dark=NAVY_TINT_2,
    table_border_color=SURFACE_COOL,
    table_border_color_dark=BORDER_ON_DARK,
    table_radius="4px",
    # Errors: the design system's hard-alert red, used only for real errors.
    error_text_color=ALERT_RED,
    error_border_color=ALERT_RED,
    error_background_fill=WHITE,
    error_background_fill_dark=NAVY_CARD,
    # Accents.
    loader_color=TEAL,
    slider_color=TEAL,
    stat_background_fill=SURFACE_PALE,
    stat_background_fill_dark=NAVY_TINT,
    code_background_fill=TABLE_ZEBRA_SCREEN,
    code_background_fill_dark=NAVY_TINT,
)

# The Proxima Fusion logo. Its wordmark is drawn in `currentColor`, so it is navy
# on the light page and white in dark mode; the mark keeps its teal.
LOGO_SVG = (Path(__file__).parent / "assets" / "logo.svg").read_text()

custom_css = """
/* Page width and base type. */
.gradio-container {
    max-width: 1280px !important;
    font-variant-numeric: tabular-nums;
}

/* Header: two-tone title top-left, logo top-right, no rule under the title. */
#pf-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 32px;
    padding: 32px 0 8px;
}
#pf-header h1 {
    margin: 0;
    font-size: 40px;
    font-weight: 500;
    line-height: 1.15;
    letter-spacing: -0.01em;
    color: var(--body-text-color);
}
#pf-header .pf-title-lead { color: #6293A0; }
#pf-header .pf-logo {
    flex: none;
    height: 36px;
    width: auto;
    color: var(--body-text-color);
}
#pf-header .pf-logo svg { height: 100%; width: auto; display: block; }

/* Prose: Medium headings, a pale-teal rule under section headers, teal dots
   for bullets. Only two weights exist, so bold renders at Medium. */
.markdown-text, .markdown-text p, .markdown-text li {
    font-size: 16px;
    line-height: 1.45;
}
.markdown-text h1, .markdown-text h2, .markdown-text h3,
.markdown-text strong, .markdown-text b, .markdown-text th {
    font-weight: 500 !important;
}
.markdown-text h2 {
    font-size: 24px;
    line-height: 1.15;
    margin: 32px 0 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(52, 206, 183, 0.55);
}
.markdown-text h3 { font-size: 20px; margin: 24px 0 8px; }
.markdown-text ul { list-style: none; padding-left: 0; }
.markdown-text ul > li {
    position: relative;
    padding-left: 22px;
    margin-bottom: 8px;
}
.markdown-text ul > li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0.6em;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #34CEB7;
}
.markdown-text table { border-collapse: collapse; width: 100%; }
.markdown-text th {
    text-align: left;
    background: var(--table-even-background-fill);
    border-bottom: 1px solid #6293A0;
    padding: 10px 14px;
}
.markdown-text td {
    border-bottom: 1px solid var(--table-border-color);
    padding: 10px 14px;
    vertical-align: top;
}
.markdown-text tbody tr:nth-child(even) td {
    background: var(--table-even-background-fill);
}
.markdown-text code {
    font-size: 0.9em;
    padding: 1px 4px;
    border-radius: 2px;
    background: var(--code-background-fill);
}

/* Tabs: Medium labels; Gradio marks the selected tab with the teal accent. */
.tab-buttons button {
    font-size: 18px;
    font-weight: 500;
}

/* Leaderboard table: Medium header with a single hairline under it. */
#leaderboard-table th {
    font-weight: 500;
    background: #E1EAF2;
    border-bottom: 1px solid #6293A0;
}
.dark #leaderboard-table th {
    background: #14233F;
}
#leaderboard-table td, #leaderboard-table th { font-size: 15px; }

/* Best value per metric column (set by the table's Styler). */
:root { --pf-best-bg: #CEDDE8; }  /* Darker than the #E1EAF2 zebra rows. */
.dark { --pf-best-bg: #143F5F; }

/* The accent (selected tab, loader) is navy on light; on navy it is teal.
   The theme has no dark variant for it, so set it here. */
.dark { --color-accent: #34CEB7 !important; }

/* Focus: 2px outline with 2px offset; navy on light, teal on navy. */
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid #00102E;
    outline-offset: 2px;
}
.dark button:focus-visible, .dark a:focus-visible,
.dark input:focus-visible, .dark textarea:focus-visible {
    outline-color: #34CEB7;
}

/* Explore tab: Netron in a frame, edged like the other blocks. */
.pf-netron iframe {
    width: 100%;
    height: 720px;
    border: 1px solid #6293A0;
    border-radius: 4px;
}

/* Pareto plot in dark mode. Plotly can't follow the page's colour scheme, so
   the figure is drawn for light mode (see `explore/pareto.py`) and recoloured
   here: white text, navy-tint grid, and the front (the second trace) in teal.
   Plotly sets colours as inline styles, hence `!important`. */
.dark #pareto-plot .main-svg text { fill: #FFFFFF !important; }
.dark #pareto-plot .gridlayer path { stroke: #14233F !important; }
.dark #pareto-plot .point,
.dark #pareto-plot .scatterpts { stroke: #0D2447 !important; }
.dark #pareto-plot .scatterlayer .trace:nth-child(2) .js-line,
.dark #pareto-plot .legend .traces:nth-child(2) .js-line {
    stroke: #34CEB7 !important;
}
.dark #pareto-plot .scatterlayer .trace:nth-child(2) .point,
.dark #pareto-plot .legend .traces:nth-child(2) .scatterpts {
    fill: #34CEB7 !important;
}

/* Footer line, bottom left. */
#pf-footer, #pf-footer p { font-size: 12px; }
"""


def header_html(lead: str, cont: str) -> str:
    """The page header: a two-tone title (`lead` in steel-blue, `cont` in navy)
    on the left and the Proxima Fusion logo on the right."""
    return (
        '<div id="pf-header">'
        f'<h1><span class="pf-title-lead">{lead}</span> {cont}</h1>'
        f'<div class="pf-logo">{LOGO_SVG}</div>'
        "</div>"
    )
