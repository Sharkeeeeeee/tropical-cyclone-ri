"""Download the IBTrACS best-track archive (since 1980) to data/raw/.

Idempotent: skips the download if a complete local copy already exists.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import IBTRACS_CSV, IBTRACS_URL  # noqa: E402


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))

    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        print(f"already present, skipping: {dest} ({remote_size/1e6:.1f} MB)")
        return

    print(f"downloading {url}\n  -> {dest} ({remote_size/1e6:.1f} MB)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    got = 0
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                got += len(block)
                if remote_size:
                    pct = 100 * got / remote_size
                    print(f"\r  {got/1e6:7.1f} / {remote_size/1e6:.1f} MB  ({pct:5.1f}%)",
                          end="", flush=True)
    print()

    if remote_size and got != remote_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch: got {got}, expected {remote_size}")

    tmp.replace(dest)
    print(f"done: {dest}")


if __name__ == "__main__":
    download(IBTRACS_URL, IBTRACS_CSV)
