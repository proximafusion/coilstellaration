import gradio as gr

from leaderboard.src.explore import tab as explore


def test_explore_tab_publishes_named_endpoints_only():
    with gr.Blocks() as demo:
        explore.render([])
    info = demo.get_api_info(all_endpoints=True)
    # The first-visit loaders are UI-only; nothing is published unnamed.
    assert info["unnamed_endpoints"] == {}
    endpoints = {
        name: (
            [p["parameter_name"] for p in api["parameters"]],
            [r["label"] for r in api["returns"]],
        )
        for name, api in info["named_endpoints"].items()
    }
    assert endpoints == {
        "/pareto": (
            ["score", "cost", "track", "stratum"],
            ["Pareto front", "Notes"],
        ),
        "/coilset_instances": (["result", "track"], ["Instance"]),
        "/coilset": (["result", "track", "instance"], ["Coilset", "Summary"]),
        "/model_graph": (["result"], ["Model graph"]),
    }
