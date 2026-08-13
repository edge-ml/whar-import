"""Live import progress helpers.

Two things the whar_datasets library does not expose directly, recovered here so
the UI can show real movement instead of an opaque spinner:

  - capture_tqdm: the library wraps its parsing/session-building loops in tqdm
    with known totals, so patching tqdm in place lets us report a genuine
    percentage during the (otherwise silent) processing phase.
  - DownloadWatcher: the download step streams bytes without surfacing progress,
    so we poll the growth of the cache directory on a background thread to show
    "X MB downloaded so far". Best-effort: degrades to nothing (elapsed only) if
    the directory can't be sized.
"""
import os
import threading
import time
from contextlib import contextmanager


@contextmanager
def capture_tqdm(report):
    """Patch tqdm in place so every counted loop reports (desc, current, total)
    via `report`. Patching the class methods (not rebinding the name) works even
    though the library already did `from tqdm import tqdm`, because every such
    reference points at the same class object. Restored on exit."""
    try:
        from tqdm import std as tqdm_std
    except Exception:
        yield
        return

    Tqdm = tqdm_std.tqdm
    orig_init = Tqdm.__init__
    orig_update = Tqdm.update
    orig_iter = Tqdm.__iter__
    last = {"t": 0.0}

    def emit(bar):
        now = time.time()
        if now - last["t"] < 0.2:  # throttle tight loops
            return
        last["t"] = now
        try:
            report(getattr(bar, "desc", None), getattr(bar, "n", None), getattr(bar, "total", None))
        except Exception:
            pass

    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        try:
            report(getattr(self, "desc", None), 0, getattr(self, "total", None))
        except Exception:
            pass

    def patched_update(self, n=1):
        r = orig_update(self, n)
        emit(self)
        return r

    def patched_iter(self):
        for obj in orig_iter(self):
            emit(self)
            yield obj

    Tqdm.__init__ = patched_init
    Tqdm.update = patched_update
    Tqdm.__iter__ = patched_iter
    try:
        yield
    finally:
        Tqdm.__init__ = orig_init
        Tqdm.update = orig_update
        Tqdm.__iter__ = orig_iter


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class DownloadWatcher:
    """Report bytes written into `path` since start, on a background thread."""

    def __init__(self, path: str, report, interval: float = 1.0):
        self._path = path
        self._report = report
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._base = 0

    def start(self):
        try:
            self._base = _dir_size(self._path)
        except Exception:
            self._base = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                self._report(max(0, _dir_size(self._path) - self._base))
            except Exception:
                pass

    def stop(self):
        self._stop.set()
