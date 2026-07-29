FROM python:3.11-bookworm
WORKDIR /app

# Install the CPU build of torch first so pip does not pull the multi-GB CUDA
# wheel when whar-datasets requires torch (mirrors the ml service Dockerfile).
# Keep this version in sync with whar-datasets' torch pin.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# whar-datasets falls back to a headless Chromium (Playwright) for datasets whose
# sources sit behind interstitial pages. Install the browser + its system deps.
RUN python -m playwright install --with-deps chromium

COPY app ./app

# Cache for downloaded/parsed datasets (mount a volume here in compose).
ENV WHAR_CACHE_DIR=/data/whar
# Base URL of the Dataset-store, including its path prefix, set at deploy time,
# e.g. http://explorer-dataset-store:3004  or  https://beta.edge-ml.org/ds
ENV DATASET_STORE_URL=""

CMD ["python", "-m", "app.main"]
