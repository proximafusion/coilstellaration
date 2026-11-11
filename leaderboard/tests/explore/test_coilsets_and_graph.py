import numpy as np
import pandas as pd

from leaderboard.src.explore import coilsets, model_graph


def _coilset_json():
    from coilstellaration import types

    n_modes = 3
    coilset = types.Coilset(
        coil_x_n=np.zeros((2, n_modes)),
        coil_y_n=np.zeros((2, n_modes)),
        coil_z_n=np.zeros((2, n_modes)),
        currents=np.ones(2),
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )
    return coilset.model_dump_json()


def _parquet(tmp_path):
    violations = {
        f"violation/{name}": [0.0, 0.2, 0.0] for name in coilsets.REQUIREMENTS
    }
    df = pd.DataFrame(
        {
            "track": ["fixed_shape", "fixed_shape", "variable_shape"],
            "stratum": ["loose", "tight", "loose"],
            "boundary_id": ["B1", "B1", "B2"],
            "score": [0.2, 0.9, 0.5],
            "strictly_feasible": [True, False, True],
            **violations,
            "coilset_json": [_coilset_json(), None, _coilset_json()],
        }
    )
    path = tmp_path / "result.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def test_instances_are_the_tracks_rows_best_first(tmp_path):
    path = _parquet(tmp_path)
    df = coilsets.instances(path, "fixed_shape")
    # Indexed by row in the file, so the same boundary's rows stay distinct.
    assert list(df.index) == [1, 0]
    assert "coilset_json" not in df.columns
    label = coilsets.instance_label(1, df.loc[1])
    assert label == "0.900 · infeasible · tight · B1 (#1)"


def test_load_coilset_reads_one_row(tmp_path):
    path = _parquet(tmp_path)
    assert coilsets.load_coilset(path, 0).n_field_periods == 3
    assert coilsets.load_coilset(path, 1) is None


def test_summary_marks_met_requirements(tmp_path):
    df = coilsets.instances(_parquet(tmp_path), "fixed_shape")
    summary = coilsets.summary_markdown(df.loc[1])
    assert "not strictly feasible" in summary
    assert "| Field error | 0.2 |" in summary
    summary = coilsets.summary_markdown(df.loc[0])
    assert "| Field error | met |" in summary


def test_netron_loads_the_stored_onnx_copy():
    raw = "org/model_abc_net.parquet"
    assert model_graph.onnx_url(raw).endswith("/resolve/main/org/model_abc_net.onnx")
    url = model_graph.netron_url(raw)
    assert url.startswith("https://netron.app/?url=https://huggingface.co/datasets/")
    html = model_graph.viewer_html(raw)
    assert "<iframe" in html
    assert "Open in Netron" in html
