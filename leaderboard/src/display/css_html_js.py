import gradio as gr

# Shared by the leaderboard and the evaluator status page. Gradio 6 takes the theme
# and CSS as `launch()` arguments, not `gr.Blocks()` ones.
theme = gr.themes.Soft(primary_hue="indigo")

custom_css = """
#space-title {
    text-align: center;
}

.markdown-text {
    font-size: 16px !important;
}

.tab-buttons button {
    font-size: 20px;
}
"""
