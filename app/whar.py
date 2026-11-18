"""Thin wrapper over the whar_datasets library (used as-is, unmodified).

Isolates the two things we need from it:
  - list the benchmark datasets with the metadata and the import settings the
    UI offers (channels, activities, sampling rate),
  - run download+parse for one dataset and hand back the raw sessions + labels,
    with those settings applied.
"""
from pathlib import Path
from typing import List, Optional

from whar_datasets.config.getter import (
    BENCHMARK_DATASET_IDS,
    WHARDatasetID,
    get_dataset_cfg,
)
from whar_datasets.processing.pipeline_pre import PreProcessingPipeline
from whar_datasets.processing.utils.resampling import (
    get_effective_sampling_freq,
    resample,
)
from whar_datasets.utils.loading import load_sessions

from app.progress import capture_tqdm


def default_rate(cfg) -> float:
    """The rate the library itself uses: the configured resampling rate, or the
    dataset's rate. Datasets whose sessions differ in rate (per_session, e.g.
    HHAR) have no single rate and always name a resampling rate."""
    return get_effective_sampling_freq(
        getattr(cfg, "sampling_freq", None), getattr(cfg, "resampling_freq", None)
    )


def _standardize_rate(sessions: dict, target_freq: float, source_freq: Optional[float]) -> dict:
    """Resample every session to one rate, with the library's own resampler.

    load_sessions returns sessions at whatever rate the parser produced, which
    is not guaranteed uniform (and per_session datasets differ by design).
    Forcing them onto one rate keeps the downstream fixed-dt time axis correct.
    """
    out = {}
    for session_id, df in sessions.items():
        if df is None or len(df) == 0:
            continue
        try:
            out[session_id] = resample(df, target_freq, source_freq=source_freq)
        except Exception as e:  # keep the native session rather than drop the data
            print(f"resample failed for session {session_id}, keeping native rate: {e}")
            out[session_id] = df
    return out


def _needs_credentials(download_url) -> bool:
    """Kaggle-hosted datasets require an API token; flag them so the UI can
    disable them until credentials are provisioned (skipped for the first cut)."""
    urls = download_url if isinstance(download_url, (list, tuple)) else [download_url]
    return any("kaggle" in str(u).lower() for u in urls)


def describe(cfg) -> dict:
    """What the UI shows about a dataset, and what can be chosen when importing."""
    return {
        "id": cfg.dataset_id,
        "name": cfg.dataset_id,
        "url": getattr(cfg, "dataset_url", None),
        "num_of_subjects": getattr(cfg, "num_of_subjects", None),
        "num_of_activities": getattr(cfg, "num_of_activities", None),
        "num_of_channels": getattr(cfg, "num_of_channels", None),
        # None for datasets whose sessions are recorded at different rates
        "sampling_freq": getattr(cfg, "sampling_freq", None),
        "default_sampling_freq": default_rate(cfg),
        "channels": list(getattr(cfg, "available_channels", []) or []),
        "activities": list(getattr(cfg, "available_activities", []) or []),
        "needs_credentials": _needs_credentials(getattr(cfg, "download_url", "")),
    }


def list_datasets(datasets_dir: str) -> List[dict]:
    """Metadata for the benchmark datasets (no download performed)."""
    out = []
    for ds_id in BENCHMARK_DATASET_IDS:
        cfg = get_dataset_cfg(ds_id, datasets_dir=datasets_dir)
        out.append({**describe(cfg), "id": ds_id.value, "name": ds_id.value})
    return out


def get_config(dataset_id: str, datasets_dir: str):
    """The library's configuration of one dataset; ValueError if unknown."""
    return get_dataset_cfg(WHARDatasetID(dataset_id), datasets_dir=datasets_dir)


def preprocess_and_load(dataset_id: str, datasets_dir: str, on_tqdm=None, sampling_freq: Optional[float] = None) -> dict:
    """Download + parse the dataset (cached under datasets_dir) and return the
    raw per-session data plus the label metadata needed for conversion,
    resampled to `sampling_freq` (default: the library's rate for it).

    Windowing output from the pipeline is ignored on purpose: edge-ml does its
    own windowing at training time, so we only need the raw sessions. Channel
    and activity choices are applied by the conversion, so the library's cache
    is the same whatever is chosen.

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
    rate = float(sampling_freq or default_rate(cfg))
    sessions = _standardize_rate(sessions, rate, getattr(cfg, "sampling_freq", None))
    return {
        "dataset_name": ds_id.value,
        "sessions": sessions,
        "session_df": session_df,
        "activity_df": activity_df,
        "sampling_freq": rate,
    }
