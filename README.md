# whar-import

Imports standard WHAR (Wearable Human Activity Recognition) datasets into edge-ml.

It wraps the [`whar-datasets`](https://github.com/teco-kit/whar-datasets) library
(download + parse the 30 benchmark datasets), converts each dataset into edge-ml's
format (one dataset per subject, one timeseries per channel, activity labelings), and
pushes the result to the Dataset-store over its HTTP API.

It runs as its own service because `whar-datasets` pulls torch + Playwright and pins
versions that conflict with the `ml` service; keeping it isolated avoids that clash.

## API

```
GET  /datasets                  list the benchmark datasets + metadata
POST /import   {dataset_id}      start an async import job -> {job_id}
GET  /import/{job_id}/status     poll job progress
```

`POST /import` forwards the caller's `jwt` cookie and `project` header to the
Dataset-store, so imported datasets land in the caller's project under their account.

## Config

| env | default | meaning |
| --- | --- | --- |
| `DATASET_STORE_URL` | (required) | Dataset-store base incl. path prefix, e.g. `http://dataset-store:3004/ds` |
| `WHAR_CACHE_DIR` | `/data/whar` | where downloaded/parsed datasets are cached (mount a volume) |

## Run

```
docker build -t whar-import .
docker run -p 3006:3006 -e DATASET_STORE_URL=... -v whar-cache:/data/whar whar-import
```

Behind Caddy the service is reached under `/whar*`. Kaggle-hosted datasets need an API
token and are flagged `needs_credentials` in `GET /datasets`; they are skipped for now.

## Tests

```
pip install pytest
pytest
```

`tests/test_conversion.py` covers the conversion core (no network).
