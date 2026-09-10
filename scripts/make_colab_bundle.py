"""Pack everything the extraction needs into one small zip for a cloud runner.

The point is bandwidth, not convenience. Extracting the North Atlantic and East
Pacific archive means pulling ~875 GB of whole GridSat files, because irwin_cdr
is stored as one compressed chunk per timestep and no client-side trick can
fetch less. On a shared or metered home connection that volume is a problem in
itself, quite apart from the twelve hours it takes.

Running the same extraction on a cloud machine moves that 875 GB onto the cloud
provider's network, next to the AWS Open Data mirror of the archive. Only the
finished patches - about 1.9 GB for the whole of NA+EP - ever travel home.

    python scripts/make_colab_bundle.py
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab_bundle.zip"

INCLUDE = [
    "src/tcri/__init__.py",
    "src/tcri/config.py",
    "src/tcri/gridsat.py",
    "src/tcri/ibtracs.py",
    # The extractor needs 3-hourly storm positions for the pre-t0 frames, and
    # those live in the raw IBTrACS file, not in the 6-hourly label table. The
    # downloader is shipped so the cloud run fetches the same pinned version
    # rather than a second, divergent code path.
    "scripts/01_download_ibtracs.py",
    "scripts/07_extract_patches.py",
    "data/processed/ri_labels.csv",
    "COLAB.md",
]


def main():
    missing = [p for p in INCLUDE if not (ROOT / p).exists()]
    if missing:
        print("missing, run the earlier pipeline steps first:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            z.write(ROOT / p, p)
    total = sum((ROOT / p).stat().st_size for p in INCLUDE)
    print(f"{OUT}  ({OUT.stat().st_size/1e6:.1f} MB zipped, "
          f"{total/1e6:.1f} MB raw)")
    for p in INCLUDE:
        print(f"  {(ROOT/p).stat().st_size/1e6:8.2f} MB  {p}")
    print("\nUpload this to the cloud runner and follow COLAB.md.")


if __name__ == "__main__":
    main()
