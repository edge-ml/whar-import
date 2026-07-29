"""Push converted datasets into the edge-ml Dataset-store over its HTTP API.

Reuses the existing endpoints (no Dataset-store changes):
  POST {base}/labelings/  -> create (idempotent by name) the "activity" labeling
  POST {base}/datasets/   -> create one dataset (addDataset path)

Auth is forwarded from the user who triggered the import: the Dataset-store
requires a signed `jwt` cookie plus a `project` header, so we pass both through.
`base` is configured to point at the Dataset-store (including whatever path
prefix it is served under, e.g. ".../ds").
"""
import hashlib
from typing import Dict, List, Tuple

import requests

_TIMEOUT = 120


def _color_for(name: str) -> str:
    """Deterministic #rrggbb from the activity name (stable across runs)."""
    return "#" + hashlib.md5(name.encode()).hexdigest()[:6]


def create_activity_labeling(
    base: str, project: str, jwt: str, labeling_name: str, activities: List[str]
) -> Tuple[str, Dict[str, str]]:
    """Create/merge the labeling and return (labeling_id, {activity_name: label_id})."""
    body = {
        "name": labeling_name,
        "labels": [{"name": a, "color": _color_for(a)} for a in activities],
    }
    r = requests.post(
        f"{base}/labelings/",
        json=body,
        headers={"project": project},
        cookies={"jwt": jwt},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    labeling = r.json()
    label_id_by_name = {lbl["name"]: lbl["_id"] for lbl in labeling["labels"]}
    return labeling["_id"], label_id_by_name


def create_dataset(base: str, project: str, jwt: str, dataset_body: dict) -> dict:
    """POST one dataset (timeSeries with [t_ms, value] data + linked labelings)."""
    r = requests.post(
        f"{base}/datasets/",
        json=dataset_body,
        headers={"project": project},
        cookies={"jwt": jwt},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def build_dataset_body(subject: dict, labeling_id: str, label_id_by_name: Dict[str, str]) -> dict:
    """Turn a conversion `subject` entry into the Dataset-store addDataset body:
    map each activity interval to a DatasetLabel {type: label_id, start, end}."""
    labels = [
        {
            "type": label_id_by_name[iv["activity_name"]],
            "start": iv["start"],
            "end": iv["end"],
        }
        for iv in subject["intervals"]
    ]
    return {
        "name": subject["name"],
        "metaData": subject["metaData"],
        "timeSeries": subject["timeSeries"],
        "labelings": [{"labelingId": labeling_id, "labels": labels}],
    }
