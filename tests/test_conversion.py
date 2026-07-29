"""Unit test for the WHAR->edge-ml conversion (pure pandas/numpy, no whar/torch)."""
import numpy as np
import pandas as pd

from app.conversion import build_conversion


def _session(n, channels, base):
    data = {"timestamp": pd.to_datetime(np.arange(n), unit="ms")}
    for j, ch in enumerate(channels):
        data[ch] = (np.arange(n) + base + j).astype(np.float32)
    return pd.DataFrame(data)


def test_build_conversion_per_subject():
    channels = ["accel_x", "accel_y", "accel_z"]
    # 2 subjects, 2 sessions each; subject 1 does activities 0 then 1, subject 2 does 1 then 0
    sessions = {
        0: _session(10, channels, 0),   # subj 1, activity 0
        1: _session(15, channels, 100),  # subj 1, activity 1
        2: _session(12, channels, 200),  # subj 2, activity 1
        3: _session(8, channels, 300),   # subj 2, activity 0
    }
    session_df = pd.DataFrame(
        {
            "session_id": [0, 1, 2, 3],
            "subject_id": [1, 1, 2, 2],
            "activity_id": [0, 1, 1, 0],
        }
    )
    activity_df = pd.DataFrame(
        {"activity_id": [0, 1], "activity_name": ["still", "walking"]}
    )

    out = build_conversion("WISDM", sessions, session_df, activity_df, sampling_freq=50.0)

    assert out["labeling_name"] == "activity"
    assert out["activities"] == ["still", "walking"]
    assert set(out["subjects"].keys()) == {1, 2}

    s1 = out["subjects"][1]
    assert s1["name"] == "WISDM - subject 1"
    assert s1["metaData"]["source"] == "whar" and s1["metaData"]["subject_id"] == "1"
    # one timeSeries per channel
    assert [ts["name"] for ts in s1["timeSeries"]] == channels
    # each channel holds all of subject 1's samples (10 + 15)
    for ts in s1["timeSeries"]:
        assert len(ts["data"]) == 25
        # [t_ms, value] pairs, times strictly increasing
        times = [pt[0] for pt in ts["data"]]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert all(isinstance(pt[1], float) for pt in ts["data"])
    # 2 sessions -> 2 label intervals, correct activity names, within the time span
    assert [iv["activity_name"] for iv in s1["intervals"]] == ["still", "walking"]
    dt = int(round(1000.0 / 50.0))  # 20 ms
    assert s1["intervals"][0]["start"] == 0
    assert s1["intervals"][0]["end"] == 9 * dt
    # second session starts one dt after the first ends
    assert s1["intervals"][1]["start"] == 9 * dt + dt

    # subject 2: sessions ordered by session_id (2 then 3) -> walking then still
    s2 = out["subjects"][2]
    assert [iv["activity_name"] for iv in s2["intervals"]] == ["walking", "still"]
    for ts in s2["timeSeries"]:
        assert len(ts["data"]) == 20  # 12 + 8

    print("conversion OK: 2 subjects, per-channel timeseries, monotonic ms axis, correct intervals")
