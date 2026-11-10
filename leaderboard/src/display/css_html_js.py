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
TABLE_ZEBRA = "#F7FAFB"
TABLE_RULE = "#EEF2F4"
BORDER_SUBTLE = "rgba(0, 16, 46, 0.12)"
ALERT_RED = "#CC0000"

# Dark mode: navy page, white text. Pre-blended navy tints (white at 4% and 8%
# over navy) stand in for the light surfaces.
NAVY_TINT = "#0A1A36"
NAVY_TINT_2 = "#14233F"
BORDER_ON_DARK = "rgba(255, 255, 255, 0.14)"

_FONT = (gr.themes.GoogleFont("Inter Tight"), "system-ui", "sans-serif")

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
    block_background_fill_dark=NAVY,
    block_border_color=BORDER_SUBTLE,
    block_border_color_dark=BORDER_ON_DARK,
    block_shadow="none",
    block_shadow_dark="none",
    block_radius="4px",
    panel_background_fill=WHITE,
    panel_background_fill_dark=NAVY,
    panel_border_color=BORDER_SUBTLE,
    panel_border_color_dark=BORDER_ON_DARK,
    border_color_primary=BORDER_SUBTLE,
    border_color_primary_dark=BORDER_ON_DARK,
    border_color_accent=TEAL,
    border_color_accent_dark=TEAL,
    border_color_accent_subdued=TEAL,
    border_color_accent_subdued_dark=TEAL,
    color_accent=TEAL,
    color_accent_soft=SURFACE_PALE,
    color_accent_soft_dark=NAVY_TINT_2,
    # Labels and titles: navy (white on dark), Medium weight.
    block_label_background_fill=WHITE,
    block_label_background_fill_dark=NAVY,
    block_label_text_color=NAVY,
    block_label_text_color_dark=WHITE,
    block_label_text_weight="500",
    block_label_border_color=BORDER_SUBTLE,
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
    input_background_fill_dark=NAVY_TINT,
    input_background_fill_focus=WHITE,
    input_background_fill_focus_dark=NAVY_TINT,
    input_border_color=SURFACE_COOL,
    input_border_color_dark=BORDER_ON_DARK,
    input_border_color_focus=TEAL,
    input_border_color_focus_dark=TEAL,
    input_placeholder_color=STEEL_BLUE,
    input_placeholder_color_dark=SURFACE_COOL,
    input_shadow="none",
    input_shadow_dark="none",
    input_shadow_focus="none",
    input_shadow_focus_dark="none",
    input_radius="4px",
    # Radios and checkboxes: pale chips; the selected one is navy.
    checkbox_label_background_fill=SURFACE_PALE,
    checkbox_label_background_fill_dark=NAVY_TINT,
    checkbox_label_background_fill_hover=SURFACE_COOL,
    checkbox_label_background_fill_hover_dark=NAVY_TINT_2,
    checkbox_label_background_fill_selected=NAVY,
    checkbox_label_background_fill_selected_dark=WHITE,
    checkbox_label_text_color=NAVY,
    checkbox_label_text_color_dark=WHITE,
    checkbox_label_text_color_selected=WHITE,
    checkbox_label_text_color_selected_dark=NAVY,
    checkbox_label_border_color=SURFACE_PALE,
    checkbox_label_border_color_dark=BORDER_ON_DARK,
    checkbox_label_border_color_hover=SURFACE_COOL,
    checkbox_label_border_color_hover_dark=BORDER_ON_DARK,
    checkbox_label_border_color_selected=NAVY,
    checkbox_label_border_color_selected_dark=WHITE,
    checkbox_label_shadow="none",
    checkbox_label_text_weight="400",
    checkbox_background_color=WHITE,
    checkbox_background_color_dark=NAVY,
    checkbox_background_color_selected=NAVY,
    checkbox_background_color_selected_dark=TEAL,
    checkbox_border_color=STEEL_BLUE,
    checkbox_border_color_dark=SURFACE_COOL,
    checkbox_border_color_selected=NAVY,
    checkbox_border_color_selected_dark=TEAL,
    checkbox_border_color_focus=TEAL,
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
    button_secondary_background_fill=SURFACE_PALE,
    button_secondary_background_fill_dark=NAVY_TINT,
    button_secondary_background_fill_hover=SURFACE_COOL,
    button_secondary_background_fill_hover_dark=NAVY_TINT_2,
    button_secondary_text_color=NAVY,
    button_secondary_text_color_dark=WHITE,
    button_secondary_text_color_hover=NAVY,
    button_secondary_text_color_hover_dark=WHITE,
    button_secondary_border_color=SURFACE_PALE,
    button_secondary_border_color_dark=BORDER_ON_DARK,
    button_secondary_border_color_hover=SURFACE_COOL,
    button_secondary_border_color_hover_dark=BORDER_ON_DARK,
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
    table_odd_background_fill_dark=NAVY,
    table_even_background_fill=TABLE_ZEBRA,
    table_even_background_fill_dark=NAVY_TINT,
    table_border_color=TABLE_RULE,
    table_border_color_dark=BORDER_ON_DARK,
    table_radius="4px",
    # Errors: the design system's hard-alert red, used only for real errors.
    error_text_color=ALERT_RED,
    error_border_color=ALERT_RED,
    error_background_fill=WHITE,
    # Accents.
    loader_color=TEAL,
    slider_color=TEAL,
    stat_background_fill=SURFACE_PALE,
    stat_background_fill_dark=NAVY_TINT,
    code_background_fill=SURFACE_PALE,
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
    border-bottom: 1px solid #E2E8EC;
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
    border-bottom: 1px solid #E2E8EC;
}
#leaderboard-table td, #leaderboard-table th { font-size: 15px; }

/* Best value per metric column (set by the table's Styler). */
:root { --pf-best-bg: #E1EAF2; }
.dark { --pf-best-bg: #143F5F; }

/* Focus: 2px teal outline with 2px offset. */
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid #34CEB7;
    outline-offset: 2px;
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
