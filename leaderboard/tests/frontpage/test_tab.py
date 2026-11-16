import gradio as gr

from leaderboard.src.frontpage import pareto
from leaderboard.src.frontpage import tab as frontpage


def _record(onnx_path, mean, strict, time):
    track = {
        "mean": mean,
        "strict_feasibility": strict,
        "inference_time": {"mean": time},
        "n": 10,
        "strata": {"tight": {"mean": mean, "strict_feasibility": strict}},
    }
    return {
        "model": "org/models",
        "onnx_path": onnx_path,
        "submitter": "alice",
        "tracks": {"variable_shape": track, "fixed_shape": {}},
    }


def test_front_page_publishes_one_pareto_endpoint():
    with gr.Blocks() as demo:
        frontpage.render(list)
    info = demo.get_api_info(all_endpoints=True)
    # Switching the cost axis, which clears the budget, is UI-only.
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
            ["score", "cost", "track", "stratum", "budget", "search", "columns"],
            ["Pareto front", "Notes", "Pareto ranking", "Download CSV"],
        ),
    }


def test_styled_table_bolds_front_one_and_mutes_unranked_and_over_budget():
    records = [
        _record("fast.onnx", 0.2, 0.2, 1e-4),
        _record("slow.onnx", 0.9, 0.1, 1.0),
        _record("beaten.onnx", 0.1, 0.1, 1e-3),
        _record("old.onnx", 0.5, 0.5, None),
    ]
    ranked = pareto.rank(
        pareto.view_frame(records, "variable_shape", "all"), "mean", "inference time"
    )
    columns = [pareto.FRONT, "onnx file", "mean", "time (ms)"]
    # 0.5 ms: slow (1000 ms) is over budget.
    styled = frontpage.styled_table(ranked, columns, "inference time", 0.5)
    shown = styled.data  # pyright: ignore[reportAttributeAccessIssue]
    assert list(shown["onnx file"]) == [
        "slow.onnx",
        "fast.onnx",
        "beaten.onnx",
        "old.onnx",
    ]
    assert list(shown[pareto.FRONT]) == ["1", "1", "2", frontpage.UNRANKED]
    ctx = styled._compute().ctx  # pyright: ignore[reportAttributeAccessIssue]

    def row_props(row):
        return {p for (r, _), props in ctx.items() if r == row for p in props}

    assert ("font-weight", "600") in row_props(0)
    assert ("color", "var(--pf-muted)") in row_props(0)  # over budget
    assert ("font-weight", "600") in row_props(1)
    assert ("color", "var(--pf-muted)") not in row_props(1)
    assert ("font-weight", "600") not in row_props(2)  # front 2
    assert ("color", "var(--pf-muted)") in row_props(3)  # unranked
