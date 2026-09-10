# Running the patch extraction on Colab

## Why

Extracting the North Atlantic + East Pacific archive means pulling **~875 GB** of
whole GridSat files. That is not a design choice: `irwin_cdr` is stored as a
single compressed chunk covering the entire globe per timestep
(`chunksizes 1 × 2000 × 5143`), so no client-side trick — HTTP range requests
included — can fetch less than the whole slab. Server-side subsetting via
OPeNDAP avoids it and costs ~2 GB, but NCEI's OPeNDAP endpoint is not always up.

On a shared, rented, or metered connection, 875 GB in a day is a problem on its
own. Running the extraction on Colab moves that traffic onto Google's network,
next to the [AWS Open Data mirror](https://registry.opendata.aws/noaa-cdr-gridsat-b1/)
of the same archive. **Only the finished patches come home: ~1.9 GB for all of
NA+EP, or ~250 MB per three-season shard.**

## Before you start

```bash
python scripts/make_colab_bundle.py
```

That writes `colab_bundle.zip` (~4 MB) containing the extraction code and the RI
label table. The labels are included rather than rebuilt in the cloud so that
the cloud run and your local runs use byte-identical labels.

## Work in shards

Free Colab sessions end after roughly 12 hours and can drop sooner. The
extraction checkpoints every 200 files, but the local disk of a dead VM is gone,
so the safe unit of work is a **shard of three seasons** written straight to
Google Drive. If a session dies you lose at most one shard, and re-running with
`--resume` picks up from its last checkpoint.

Measured on a free Colab VM with four workers: **2.0 files/s**. (A two-worker
smoke test gives 0.6, so do not extrapolate the estimate from that; the extra
workers help more than the two-core count suggests, because most of each file's
time is spent waiting on the network rather than decompressing.)

The North Atlantic plus East Pacific archive needs 18,832 distinct scene times:

| shard | scene times | at 2.0 files/s | cumulative |
|---|---|---|---|
| recon_2000_2002 | 2,406 | 20 min | 0.3 h |
| recon_2003_2005 | 2,608 | 22 min | 0.7 h |
| recon_2006_2008 | 2,114 | 18 min | 1.0 h |
| recon_2009_2011 | 2,150 | 18 min | 1.3 h |
| recon_2012_2014 | 2,580 | 22 min | 1.6 h |
| recon_2015_2017 | 2,774 | 23 min | 2.0 h |
| recon_2018_2020 | 2,606 | 22 min | 2.4 h |
| recon_2021_2022 | 1,594 | 13 min | 2.6 h |
| **total** | **18,832** | | **~2.6 h** |

That fits inside a single free session with room to spare. If a session does die,
note that only completed shards are on Drive: the in-progress one restarts from
zero, because its checkpoint lives on the VM's local disk. The most a disconnect
can cost is one shard, about 20 minutes.

**A display quirk, not a fault.** The extractor writes progress with a leading
carriage return so a terminal can overwrite the line in place. Read through a
pipe, that means each progress line is only delimited when the *next* one is
written, so the first visible line appears after 100 files rather than 50, and
every line lags one step behind. Nothing is wrong if the cell shows only the
`source=s3:` banner for the first couple of minutes.

## Cells

**1 — mount Drive and unpack**

```python
from google.colab import drive, files
import os, shutil

drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/tcri/patches', exist_ok=True)

shutil.rmtree('/content/work', ignore_errors=True)
os.makedirs('/content/work', exist_ok=True)
up = files.upload()                      # choose colab_bundle.zip
name = next(iter(up))                    # whatever the browser called it
shutil.unpack_archive(name, '/content/work')

# Fail loudly here rather than three hours later. files.upload() drops the file
# in the notebook's working directory, not in /content/work, so unzipping from
# the wrong place leaves an empty tree that every later cell silently tolerates.
assert os.path.exists('/content/work/scripts/07_extract_patches.py'), 'unpack failed'
assert os.path.exists('/content/work/data/processed/ri_labels.csv'), 'labels missing'
print('ok:', sorted(os.listdir('/content/work')))
```

**2 — check the environment**

```python
!pip -q install netCDF4 xarray
!nproc && free -g | head -2
```

Note the core count. Opening and decompressing a 48 MB file is CPU work, so on a
2-core free VM more than ~4 workers will not help.

**2a — fetch IBTrACS**

The extractor needs 3-hourly storm positions for the frames before t0, and those
come from the raw IBTrACS file rather than the 6-hourly label table shipped in
the bundle. It is 144 MB and downloads in seconds on Colab's network. The
version is pinned in `config.py`, so this is the same file the labels were built
from.

```python
!cd /content/work && python -u scripts/01_download_ibtracs.py
```

**2b — smoke test, three files only**

Run this before committing to hours. It uses `!` rather than `subprocess.run`,
because Colab shows the output of a `!` command in the cell and swallows the
output of a subprocess, which is exactly how a failure can look like silence.

```python
!cd /content/work && python -u scripts/07_extract_patches.py \
    --basin NA --seasons 2020 --limit 2 --workers 2 \
    --source s3 --tag smoketest --overwrite
```

You should see `cases: 2`, then a progress line, then `array: ...`. Any
traceback here is the real error; fix it before running cell 3. Common ones:

| Message | Cause |
|---|---|
| `ModuleNotFoundError: netCDF4` | cell 2 was skipped |
| `FileNotFoundError: ri_labels.csv` | cell 1's unpack assert did not pass |
| `No module named 'tcri'` | run from `/content/work`, not `/content` |

**3 — run every shard**

One cell does all eight. Each shard is copied to Drive the moment it finishes,
and anything already in Drive is copied back first, so re-running this cell
after a disconnect picks up exactly where it stopped. Safe to run as many times
as it takes.

```python
import os, subprocess, shutil, glob
SHARDS = [(2000,2002),(2003,2005),(2006,2008),(2009,2011),
          (2012,2014),(2015,2017),(2018,2020),(2021,2022)]
WORK = "/content/work"
PATCH = f"{WORK}/data/processed/patches"
DRIVE = "/content/drive/MyDrive/tcri/patches"
os.makedirs(PATCH, exist_ok=True); os.makedirs(DRIVE, exist_ok=True)

# Restore anything a previous session already finished.
for f in glob.glob(f"{DRIVE}/*"):
    shutil.copy(f, PATCH)

for lo, hi in SHARDS:
    tag = f"recon_{lo}_{hi}"
    seasons = " ".join(str(y) for y in range(lo, hi + 1))
    print(f"=== {tag} ===", flush=True)
    # Stream the child's output into the cell. subprocess.run writes to the
    # kernel's stdout, which Colab does not display, so a crash inside the
    # extraction arrives as an unexplained exit code and nothing else.
    proc = subprocess.Popen(
        f"cd {WORK} && python -u scripts/07_extract_patches.py "
        f"--basin NA EP --seasons {seasons} "
        f"--source s3 --workers 4 --tag {tag} --resume",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-25:]
        if "files" in line or "===" in line or "Error" in line or "Traceback" in line:
            print(line, flush=True)
    r = proc.wait()
    # Never report success without checking. A failed extraction that prints
    # "saved" is worse than one that crashes, because it hides the failure.
    if r != 0:
        print("\n".join(tail))
        raise RuntimeError(f"{tag}: extraction exited {r}")
    produced = glob.glob(f"{PATCH}/patches_{tag}.npy")
    if not produced:
        raise RuntimeError(f"{tag}: extraction produced no patch array")
    n = 0
    for suffix in ("patches", "index", "nanfrac"):
        for f in glob.glob(f"{PATCH}/{suffix}_{tag}.*"):
            shutil.copy(f, DRIVE)
            n += 1
    size = os.path.getsize(produced[0]) / 1e6
    print(f"{tag}: {n} files, {size:.0f} MB -> Drive", flush=True)

print("final contents of Drive:")
for f in sorted(glob.glob(f"{DRIVE}/*")):
    print(f"  {os.path.getsize(f)/1e6:8.1f} MB  {os.path.basename(f)}")
```

If the session dies mid-shard, reconnect, re-run cells 1 and 2, then re-run this
cell. Completed shards are skipped in seconds; the interrupted one resumes from
its last checkpoint.

## If Drive will not mount

`MessageError: credential propagation was unsuccessful` is a Colab
authentication failure, not a problem with this code. Re-running the cell often
clears it; failing that, allow third-party cookies for `accounts.google.com` and
`colab.research.google.com`, and do not use a private window.

Drive is only there to survive a dead session. The alternative is to download
each shard to your own machine as it finishes, which drops the dependency
entirely and moves the same ~1.9 GB you would have downloaded from Drive anyway.

**1b — unpack, no Drive**

```python
from google.colab import files
import os, shutil

shutil.rmtree("/content/work", ignore_errors=True)
os.makedirs("/content/work", exist_ok=True)
up = files.upload()                      # choose colab_bundle.zip
name = next(iter(up))                    # whatever the browser called it
shutil.unpack_archive(name, "/content/work")
print(sorted(os.listdir("/content/work")))
```

**3b — run shards, downloading each as it completes**

```python
import os, subprocess, shutil, glob
from google.colab import files

SHARDS = [(2000,2002),(2003,2005),(2006,2008),(2009,2011),
          (2012,2014),(2015,2017),(2018,2020),(2021,2022)]
WORK  = "/content/work"
PATCH = f"{WORK}/data/processed/patches"
os.makedirs(PATCH, exist_ok=True)

done = []
for lo, hi in SHARDS:
    tag = f"recon_{lo}_{hi}"
    seasons = " ".join(str(y) for y in range(lo, hi + 1))
    print(f"=== {tag} ===", flush=True)
    r = subprocess.run(
        f"cd {WORK} && python -u scripts/07_extract_patches.py "
        f"--basin NA EP --seasons {seasons} "
        f"--source s3 --workers 4 --tag {tag} --resume",
        shell=True)
    if r.returncode != 0 or not glob.glob(f"{PATCH}/patches_{tag}.npy"):
        raise RuntimeError(f"{tag}: extraction failed, nothing produced")
    shutil.make_archive(f"/content/{tag}", "zip", root_dir=PATCH)
    files.download(f"/content/{tag}.zip")
    done.append((lo, hi))
    remaining = [x for x in SHARDS if x not in done]
    print(f"{tag} downloaded. If the session dies, set SHARDS = {remaining}")
```

The zip of each shard carries every patch file present at that moment, so the
last one downloaded contains everything. If the session dies, re-run cells 1b
and 2, set `SHARDS` to the list the last message printed, and continue: only the
interrupted shard is lost, at most 40 minutes.

## Bringing it home

Download `MyDrive/tcri/patches/` (~1.9 GB total) into
`data/processed/patches/`, then merge the shards into one array:

```bash
python scripts/13_merge_shards.py --out recon_2000_2022 --tags \
  recon_2000_2002 recon_2003_2005 recon_2006_2008 recon_2009_2011 \
  recon_2012_2014 recon_2015_2017 recon_2018_2020 recon_2021_2022
```

From there the rest of the pipeline is unchanged:

```bash
python scripts/08_validate_patches.py recon_2000_2022
python -u scripts/10_structural_features.py --tag recon_2000_2022
python -u scripts/11_structure_baseline.py --tag recon_2000_2022 --group SEASON
python -u scripts/12_train_gpu.py --tag recon_2000_2022 --group SEASON
```

## If OPeNDAP comes back

Check it with:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  'https://www.ncei.noaa.gov/thredds/dodsC/cdr/gridsat/2005/GRIDSAT-B1.2005.08.28.12.v02r01.nc.dds'
```

A `200` means you can run the whole thing at home for ~2 GB instead:

```bash
python -u scripts/07_extract_patches.py --basin NA EP --seasons $(seq 2000 2022) \
    --workers 6 --tag recon_2000_2022 --source opendap --resume
```

`--resume` records which frames are done, not which route fetched them, so the
two paths mix freely. They have been verified to produce bit-identical patches.

## Other runners

Nothing here is Colab-specific beyond the Drive cells. A cheap cloud VM in
`us-east-1` is strictly better if you have an account: S3 reads are free and
fast from inside the region, and the whole extraction finishes in an hour or two
for well under a dollar. The `data pipeline` line in the grant budget is exactly
this.
