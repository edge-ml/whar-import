"""whar-import service: imports standard WHAR datasets into edge-ml.

Runs isolated from the other services because whar_datasets pulls torch/
playwright and version-pins that conflict with the ml service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import whar as whar_router

app = FastAPI(title="edge-ml whar-import")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(whar_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # proxy_headers/forwarded_allow_ips so redirects/URLs use the external https
    # scheme behind Caddy (same fix the ml/dataset-store services needed).
    # Single worker: the in-process job registry is not shared across workers.
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=3006,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
