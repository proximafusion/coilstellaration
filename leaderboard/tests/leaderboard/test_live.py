import pandas as pd

from leaderboard.src.display.utils import COLS, STRATUM_VIEWS, TRACKS
from leaderboard.src.live import QUEUE_STATUSES, LiveData, Snapshot


def test_live_data_starts_empty_with_every_view_and_queue_table():
    snapshot = LiveData(load=Snapshot).current
    assert set(snapshot.frames) == {(t, s) for t in TRACKS for s in STRATUM_VIEWS}
    assert all(list(f.columns) == COLS for f in snapshot.frames.values())
    assert snapshot.records == []
    assert list(snapshot.queue) == list(QUEUE_STATUSES)


def test_refresh_swaps_in_the_new_snapshot():
    loaded = Snapshot(records=[{"model": "org/alpha"}])
    live = LiveData(load=lambda: loaded)
    assert live.refresh()
    assert live.current is loaded


def test_failed_refresh_keeps_the_current_snapshot():
    first = Snapshot(records=[{"model": "org/alpha"}])
    live = LiveData(load=lambda: first)
    assert live.refresh()

    def unreachable():
        raise ConnectionError("the Hub is down")

    live._load = unreachable
    assert not live.refresh()
    assert live.current is first


def test_snapshot_frames_cover_every_view_from_records():
    record = {
        "model": "org/alpha",
        "submitter": "alice",
        "raw_scores_file": "org/alpha_abc_model.parquet",
        "tracks": {"variable_shape": {"mean": 0.5, "strict_feasibility": 0.1}},
    }
    from leaderboard.src.populate import frames_from_records

    frames = frames_from_records([record], COLS)
    assert len(frames[("variable_shape", "all")]) == 1
    assert isinstance(frames[("fixed_shape", "all")], pd.DataFrame)
    assert frames[("fixed_shape", "all")].empty
