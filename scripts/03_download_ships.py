"""Download the CIRA SHIPS developmental (lsdiag) predictor files.

All seven basins are published, but with different end years and slightly
different predictor sets: the Atlantic-family files (AL/CP/EP) run to 2023 and
carry the full satellite predictor block, while the JTWC basins (IO/SH/WP) stop
at 2021. That asymmetry has to be carried through the evaluation rather than
hidden by pooling.

The server rejects non-browser user agents, hence the explicit header.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import SHIPS_FILES, SHIPS_RAW_DIR  # noqa: E402

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    head = requests.head(url, headers=HEADERS, timeout=60, allow_redirects=True)
    head.raise_for_status()
    size = int(head.headers.get("content-length", 0))

    if dest.exists() and size and dest.stat().st_size == size:
        print(f"  present, skipping ({size/1e6:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    got = 0
    with requests.get(url, headers=HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                got += len(block)
                if size:
                    print(f"\r  {got/1e6:7.1f} / {size/1e6:.1f} MB "
                          f"({100*got/size:5.1f}%)", end="", flush=True)
    print()
    if size and got != size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {url}: {got} != {size}")
    tmp.replace(dest)


def main(basins=None):
    targets = SHIPS_FILES if basins is None else {
        b: SHIPS_FILES[b] for b in basins}
    for basin, url in targets.items():
        dest = SHIPS_RAW_DIR / url.rsplit("/", 1)[1]
        print(f"[{basin}] {dest.name}")
        download(url, dest)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
