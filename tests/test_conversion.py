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


def _two_activities():
    channels = ["acc_x", "acc_y", "gyro_x"]
    sessions = {0: _session(10, channels, 0), 1: _session(10, channels, 100), 2: _session(10, channels, 200)}
    session_df = pd.DataFrame({"session_id": [0, 1, 2], "subject_id": [1, 1, 2], "activity_id": [0, 1, 1]})
    activity_df = pd.DataFrame({"activity_id": [0, 1], "activity_name": ["still", "walking"]})
    return sessions, session_df, activity_df


def test_chosen_channels_and_activities():
    sessions, session_df, activity_df = _two_activities()
    out = build_conversion("X", sessions, session_df, activity_df, 50.0,
                           channels=["acc_x", "gyro_x"], activities=["walking"])
    assert out["activities"] == ["walking"]
    assert set(out["subjects"]) == {1, 2}
    subject = out["subjects"][1]
    assert [ts["name"] for ts in subject["timeSeries"]] == ["acc_x", "gyro_x"]
    # only the walking session of subject 1 (values from base 100)
    assert subject["timeSeries"][0]["data"][0][1] == 100.0
    assert len(subject["timeSeries"][0]["data"]) == 10
    assert [i["activity_name"] for i in subject["intervals"]] == ["walking"]


def test_a_subject_without_chosen_activities_is_left_out():
    sessions, session_df, activity_df = _two_activities()
    out = build_conversion("X", sessions, session_df, activity_df, 50.0, activities=["still"])
    assert set(out["subjects"]) == {1}


def test_unknown_choices_are_refused():
    sessions, session_df, activity_df = _two_activities()
    for kwargs in ({"channels": ["mag_x"]}, {"activities": ["flying"]}):
        try:
            build_conversion("X", sessions, session_df, activity_df, 50.0, **kwargs)
        except ValueError as e:
            assert "unknown" in str(e)
        else:
            raise AssertionError("accepted an unknown choice")


def test_rates_that_do_not_divide_1000_keep_their_rate():
    sessions, session_df, activity_df = _two_activities()
    out = build_conversion("X", sessions, session_df, activity_df, 30.0)
    times = [t for t, _ in out["subjects"][1]["timeSeries"][0]["data"]]
    # 10 samples per session at 30 Hz, two sessions back to back
    assert times[:4] == [0, 33, 67, 100]
    assert times[10] == 333  # the second session starts one period after the first ends
    assert out["subjects"][1]["metaData"]["sampling_freq"] == "30"
