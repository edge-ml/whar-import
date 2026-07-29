"""HTTP API for importing standard WHAR datasets into edge-ml.

  GET  /datasets                 -> the benchmark datasets + metadata
  POST /import   {dataset_id}     -> start an async import job, returns {job_id}
  GET  /import/{job_id}/status    -> job progress

The import runs in the background (download can take minutes). Job state is kept
in-process, so run this service with a single worker. Auth (jwt cookie + project
header) is forwarded to the Dataset-store on the caller's behalf.
"""
import os
import traceback
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from app import conversion, datasetstore_client as ds, whar

router = APIRouter()

CACHE_DIR = os.environ.get("WHAR_CACHE_DIR", "/data/whar")
DATASET_STORE_BASE = os.environ.get("DATASET_STORE_URL", "").rstrip("/")

# In-process job registry (single worker). job_id -> status dict.
_jobs: Dict[str, dict] = {}


@router.get("/datasets")
def get_datasets():
    return whar.list_datasets(CACHE_DIR)


@router.post("/import")
def start_import(
    body: dict,
    background_tasks: BackgroundTasks,
    request: Request,
    project: str = Header(...),
):
    dataset_id = (body or {}).get("dataset_id")
    if not dataset_id:
        raise HTTPException(400, "dataset_id is required")
    jwt = request.cookies.get("jwt")
    if not jwt:
        raise HTTPException(401, "missing jwt cookie")
    if not DATASET_STORE_BASE:
        raise HTTPException(500, "DATASET_STORE_URL is not configured")

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "state": "queued",
        "dataset_id": dataset_id,
        "subjects_done": 0,
        "subjects_total": None,
        "created_dataset_ids": [],
        "error": None,
    }
    background_tasks.add_task(_run_import, job_id, dataset_id, project, jwt)
    return {"job_id": job_id}


@router.get("/import/{job_id}/status")
def import_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


def _run_import(job_id: str, dataset_id: str, project: str, jwt: str):
    job = _jobs[job_id]
    try:
        job["state"] = "downloading"  # whar download + parse (cached)
        loaded = whar.preprocess_and_load(dataset_id, CACHE_DIR)

        job["state"] = "converting"
        conv = conversion.build_conversion(
            loaded["dataset_name"],
            loaded["sessions"],
            loaded["session_df"],
            loaded["activity_df"],
            loaded["sampling_freq"],
        )
        subjects = conv["subjects"]
        job["subjects_total"] = len(subjects)

        job["state"] = "uploading"
        labeling_id, label_id_by_name = ds.create_activity_labeling(
            DATASET_STORE_BASE, project, jwt, conv["labeling_name"], conv["activities"]
        )
        for subject in subjects.values():
            body = ds.build_dataset_body(subject, labeling_id, label_id_by_name)
            created = ds.create_dataset(DATASET_STORE_BASE, project, jwt, body)
            job["created_dataset_ids"].append(created.get("_id"))
            job["subjects_done"] += 1

        job["state"] = "done"
    except Exception as e:  # surface a readable reason to the poller
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        print(traceback.format_exc())
