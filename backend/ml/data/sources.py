"""Raw dataset locations and a resumable in-memory fetcher.

Every loader first looks for a local copy in data/raw/ and otherwise streams
the file into memory — nothing large is written to disk, so the build works
on a nearly full drive (and on Colab without a manual download step).
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

GH = "https://raw.githubusercontent.com"
URLS = {
    "banking77_train.csv": f"{GH}/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "banking77_test.csv": f"{GH}/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    "clinc150_full.json": f"{GH}/clinc/oos-eval/master/data/data_full.json",
    **{f"snips_{s}_{f}": f"{GH}/MiuLab/SlotGated-SLU/master/data/snips/{s}/{f}"
       for s in ("train", "valid", "test") for f in ("seq.in", "label")},
    **{f"goemotions_{s}.tsv": f"{GH}/google-research/google-research/master/goemotions/data/{s}.tsv"
       for s in ("train", "dev", "test")},
    "goemotions_emotions.txt": f"{GH}/google-research/google-research/master/goemotions/data/emotions.txt",
    # Original Kaggle "Customer Support on Twitter" CSV (thoughtvector), mirrored on the HF Hub.
    "twcs.csv": "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv",
    "abcd_v1.1.json.gz": "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz",
}


def fetch_bytes(name: str, retries: int = 8) -> bytes:
    """Local data/raw/<name> if present, else download into memory,
    resuming with HTTP Range requests after a dropped connection."""
    local = RAW_DIR / name
    if local.is_file():
        return local.read_bytes()
    url, buf, total = URLS[name], bytearray(), None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={len(buf)}-"} if buf else {})
            with urllib.request.urlopen(req, timeout=60) as resp:
                if total is None:
                    total = int(resp.headers.get("Content-Length") or 0) or None
                while chunk := resp.read(1 << 20):
                    buf += chunk
            if total is None or len(buf) >= total:
                return bytes(buf)
        except Exception as exc:
            print(f"[fetch] {name}: {exc!r} after {len(buf)} bytes, retry {attempt + 1}/{retries}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {name} from {url}")


def open_text(name: str) -> io.StringIO:
    return io.StringIO(fetch_bytes(name).decode("utf-8"))


class ResumingHTTPStream(io.RawIOBase):
    """Read-once HTTP stream that reconnects with a Range header when the
    connection drops, so a long scan survives a flaky network."""

    def __init__(self, url: str, retries: int = 20):
        self.url, self.pos, self.retries, self.resp = url, 0, retries, None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        for attempt in range(self.retries):
            try:
                if self.resp is None:
                    headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
                    self.resp = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=60)
                n = self.resp.readinto(b)
                self.pos += n
                return n
            except Exception as exc:
                print(f"[stream] {exc!r} at byte {self.pos}, reconnecting ({attempt + 1}/{self.retries})")
                self.resp = None
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"stream from {self.url} failed at byte {self.pos}")

    def close(self):
        if self.resp is not None:
            self.resp.close()
        super().close()


def open_stream(name: str):
    """A buffered binary stream for large files: the local copy if present,
    else the HTTP response itself (read once, never stored)."""
    local = RAW_DIR / name
    if local.is_file():
        return local.open("rb")
    return io.BufferedReader(ResumingHTTPStream(URLS[name]), buffer_size=1 << 20)
