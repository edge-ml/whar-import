"""Convert whar-datasets output into edge-ml datasets (one per subject).

Input is what `whar_datasets` produces after ``PreProcessingPipeline.run()``:
  - sessions:   Dict[int session_id -> pd.DataFrame]  (a "timestamp" column plus
                one float column per sensor channel), from ``load_sessions``.
  - session_df: pd.DataFrame with columns session_id, subject_id, activity_id.
  - activity_df: pd.DataFrame with columns activity_id, activity_name.
  - sampling_freq: sampling rate in Hz (from the dataset config).

Output is edge-ml's ingestion shape (see Dataset-store addDataset): one dataset
per subject, each with one timeSeries per channel carrying [time_ms, value]
pairs on a shared monotonic epoch-millisecond axis, plus per-session activity
label intervals on that same axis. A single project-wide "activity" labeling is
described once (its labels are created in the Dataset-store before the datasets
are pushed, and the intervals reference them by activity name).

This module is pure (pandas/numpy only) so it can be unit-tested without the
heavy whar_datasets / torch dependencies.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LABELING_NAME = "activity"
TIMESTAMP_COL = "timestamp"


def infer_channels(sessions: Dict[int, pd.DataFrame]) -> List[str]:
    """Channel columns = every column of a session DataFrame except the timestamp."""
    for df in sessions.values():
        return [c for c in df.columns if c != TIMESTAMP_COL]
    return []


def build_conversion(
    dataset_name: str,
    sessions: Dict[int, pd.DataFrame],
    session_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    sampling_freq: float,
    channels: Optional[List[str]] = None,
) -> dict:
    """Return {labeling_name, activities, subjects} where subjects maps
    subject_id -> {name, metaData, timeSeries, intervals}. `intervals` are
    {activity_name, start, end} (epoch-ms); they are mapped to label ids when
    the dataset is pushed."""
    if channels is None:
        channels = infer_channels(sessions)
    if not channels:
        raise ValueError("no sensor channels found in sessions")
    if sampling_freq <= 0:
        raise ValueError(f"invalid sampling_freq: {sampling_freq}")

    dt_ms = max(1, int(round(1000.0 / sampling_freq)))
    activity_name_by_id = dict(
        zip(activity_df["activity_id"].tolist(), activity_df["activity_name"].tolist())
    )

    subjects: Dict[int, dict] = {}
    for subject_id, sub in session_df.groupby("subject_id"):
        # Lay this subject's sessions end-to-end on one monotonic ms axis.
        cursor = 0
        ts_data: Dict[str, List[list]] = {ch: [] for ch in channels}
        intervals: List[dict] = []

        for row in sub.sort_values("session_id").itertuples(index=False):
            sdf = sessions.get(int(row.session_id))
            if sdf is None or len(sdf) == 0:
                continue
            n = len(sdf)
            times = cursor + np.arange(n, dtype=np.int64) * dt_ms
            for ch in channels:
                col = np.asarray(sdf[ch].to_numpy(), dtype=np.float64)
                ts_data[ch].extend([int(t), float(v)] for t, v in zip(times, col))
            intervals.append(
                {
                    "activity_name": activity_name_by_id[int(row.activity_id)],
                    "start": int(times[0]),
                    "end": int(times[-1]),
                }
            )
            cursor = int(times[-1]) + dt_ms

        if not intervals:  # subject had only empty sessions
            continue

        subjects[int(subject_id)] = {
            "name": f"{dataset_name} - subject {subject_id}",
            "metaData": {
                "source": "whar",
                "whar_id": dataset_name,
                "subject_id": str(subject_id),
            },
            "timeSeries": [
                {"name": ch, "unit": "", "data": ts_data[ch]} for ch in channels
            ],
            "intervals": intervals,
        }

    # All activity names that appear anywhere, in stable activity_id order.
    activities = [
        activity_name_by_id[a]
        for a in sorted(activity_df["activity_id"].tolist())
    ]
    return {"labeling_name": LABELING_NAME, "activities": activities, "subjects": subjects}
