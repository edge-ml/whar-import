"""Thin wrapper over the whar_datasets library (used as-is, unmodified).

Isolates the two things we need from it:
  - list the benchmark datasets with the metadata the UI shows,
  - run download+parse for one dataset and hand back the raw sessions + labels.

Importing whar_datasets pulls torch (its package __init__ imports TorchAdapter),
which is why this service runs in its own image; that is expected here.
"""
from pathlib import Path
from typing import List

from whar_datasets.config.getter import (
    BENCHMARK_DATASET_IDS,
    WHARDatasetID,
    get_dataset_cfg,
)
from whar_datasets.processing.pipeline_pre import PreProcessingPipeline
from whar_datasets.utils.loading import load_sessions

from app.progress import capture_tqdm


def _needs_credentials(download_url) -> bool:
    """Kaggle-hosted datasets require an API token; flag them so the UI can
    disable them until credentials are provisioned (skipped for the first cut)."""
    urls = download_url if isinstance(download_url, (list, tuple)) else [download_url]
    return any("kaggle" in str(u).lower() for u in urls)


def list_datasets(datasets_dir: str) -> List[dict]:
    """Metadata for the 30 benchmark datasets (no download performed)."""
    out = []
    for ds_id in BENCHMARK_DATASET_IDS:
        cfg = get_dataset_cfg(ds_id, datasets_dir=datasets_dir)
        out.append(
            {
                "id": ds_id.value,
                "name": ds_id.value,
                "num_of_subjects": getattr(cfg, "num_of_subjects", None),
                "num_of_activities": getattr(cfg, "num_of_activities", None),
                "num_of_channels": getattr(cfg, "num_of_channels", None),
                "sampling_freq": getattr(cfg, "sampling_freq", None),
                "needs_credentials": _needs_credentials(getattr(cfg, "download_url", "")),
            }
        )
    return out


def preprocess_and_load(dataset_id: str, datasets_dir: str, on_tqdm=None) -> dict:
    """Download + parse the dataset (cached under datasets_dir) and return the
    raw per-session data plus the label metadata needed for conversion.

    Windowing output from the pipeline is ignored on purpose: edge-ml does its
    own windowing at training time, so we only need the raw sessions.

    on_tqdm(desc, current, total), if given, receives live progress from the
    library's internal counted loops (the processing phase), so the caller can
    surface a real percentage.
    """
    ds_id = WHARDatasetID(dataset_id)
    cfg = get_dataset_cfg(ds_id, datasets_dir=datasets_dir)
    pipe = PreProcessingPipeline(cfg)
    if on_tqdm is not None:
        with capture_tqdm(on_tqdm):
            activity_df, session_df, _window_df = pipe.run()
            sessions = load_sessions(Path(pipe.sessions_dir))
    else:
        activity_df, session_df, _window_df = pipe.run()
        sessions = load_sessions(Path(pipe.sessions_dir))
    return {
        "dataset_name": ds_id.value,
        "sessions": sessions,
        "session_df": session_df,
        "activity_df": activity_df,
        "sampling_freq": float(cfg.sampling_freq),
    }
