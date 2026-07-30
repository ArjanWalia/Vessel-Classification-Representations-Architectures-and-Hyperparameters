

import os
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/vessel_classification_final/Dataset")
DRIVE_SRC = DRIVE_ROOT / "cumulative_dataset"
DRIVE_TAR = DRIVE_ROOT / "cumulative_dataset.tar"

LOCAL_SRC = Path("/content/cumulative_dataset")
LOCAL_TAR = Path("/content/cumulative_dataset.tar")

THREADS = 16
SKIP_EXISTING = True
MAKE_TAR = True
COPY_TAR_TO_DRIVE = True



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")


def copy_one(pair):
    src, dst = pair
    try:
        if SKIP_EXISTING and dst.exists() and dst.stat().st_size > 0:
            return "skip"
        shutil.copy2(src, dst)
        return "copy"
    except Exception as exc:
        return f"fail:{src}:{exc}"


def main():
    if not DRIVE_SRC.is_dir():
        raise SystemExit(f"source not found: {DRIVE_SRC}")

    print(f"scanning {DRIVE_SRC} ...", flush=True)
    t0 = time.time()
    wavs = list(DRIVE_SRC.rglob("*.wav"))
    print(f"  found {len(wavs)} wav files in {time.time() - t0:.0f}s\n", flush=True)
    if not wavs:
        raise SystemExit("no wavs found - check the path")

    dirs = {LOCAL_SRC / p.parent.relative_to(DRIVE_SRC) for p in wavs}
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"created {len(dirs)} class folders under {LOCAL_SRC}", flush=True)

    pairs = [(p, LOCAL_SRC / p.relative_to(DRIVE_SRC)) for p in wavs]
    already = sum(1 for _, d in pairs if d.exists() and d.stat().st_size > 0)
    if already:
        print(f"{already} files already present -> resuming, "
              f"{len(pairs) - already} to go", flush=True)

    print(f"\ncopying on {THREADS} threads...", flush=True)
    t0 = time.time()
    copied = skipped = 0
    failures = []
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futs = [pool.submit(copy_one, p) for p in pairs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r == "copy":
                copied += 1
            elif r == "skip":
                skipped += 1
            else:
                failures.append(r)
            if i % 1000 == 0 or i == len(pairs):
                el = time.time() - t0
                rate = i / max(el, 1e-9)
                eta = (len(pairs) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(pairs)}  ({rate:.0f} files/s, "
                      f"eta {eta / 60:.1f} min)", flush=True)

    mins = (time.time() - t0) / 60
    print(f"\ncopied {copied}, skipped {skipped} in {mins:.1f} min", flush=True)
    if failures:
        print(f"  !! {len(failures)} failures:", flush=True)
        for f in failures[:10]:
            print(f"     {f}", flush=True)

    local_n = sum(1 for _ in LOCAL_SRC.rglob("*.wav"))
    size_mb = sum(p.stat().st_size for p in LOCAL_SRC.rglob("*.wav")) / 1e6
    print(f"\nlocal dataset: {local_n} wavs, {size_mb:.0f} MB", flush=True)
    if local_n != len(wavs):
        print(f"  !! expected {len(wavs)} - do not tar an incomplete copy", flush=True)
        return

    if MAKE_TAR:
        print(f"\ntarring -> {LOCAL_TAR} (local disk, fast)...", flush=True)
        t0 = time.time()
        with tarfile.open(LOCAL_TAR, "w") as tar:
            tar.add(LOCAL_SRC, arcname=LOCAL_SRC.name)
        print(f"  {LOCAL_TAR.stat().st_size / 1e6:.0f} MB in "
              f"{time.time() - t0:.0f}s", flush=True)

        if COPY_TAR_TO_DRIVE:
            print(f"\ncopying tar -> {DRIVE_TAR} ...", flush=True)
            t0 = time.time()
            sh(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
            print(f"  copied in {time.time() - t0:.0f}s", flush=True)
            print("\nevery grid/builder script will now find cumulative_dataset.tar", flush=True)
            print("and copy ONE file instead of 19,600 - a minute or two, not an hour.", flush=True)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
