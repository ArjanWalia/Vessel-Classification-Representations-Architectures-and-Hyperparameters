import os
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

# ----------------------------- CONFIG ---------------------------------------

DRIVE_ROOT = Path("")
DRIVE_SRC = DRIVE_ROOT / ""
DRIVE_TAR = DRIVE_ROOT / ""

LOCAL_SRC = Path("")
LOCAL_OUT = Path("")
LOCAL_TAR = Path("")

SPLITS = ["train", "validation", "test"]


SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 128           
N_MELS = 128
FMIN = 0
FMAX = SR // 2            
CLIP_SAMPLES = SR         

LOG_SCALE = True          
MINMAX_01 = True           
DTYPE = np.float32        

WORKERS = max(1, cpu_count() - 1)
STAGE_FROM_DRIVE = True    


def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")


def stage_source():
    if LOCAL_SRC.is_dir() and any(LOCAL_SRC.rglob("*.wav")):
        print(f"source already staged at {LOCAL_SRC}", flush=True)
        return
    if not STAGE_FROM_DRIVE:
        raise SystemExit(f"{LOCAL_SRC} missing and STAGE_FROM_DRIVE is False")
    if not DRIVE_SRC.is_dir():
        raise SystemExit(f"source not found on Drive: {DRIVE_SRC}")
    print(f"staging {DRIVE_SRC} -> {LOCAL_SRC} (one bulk copy, not per-file)...",
          flush=True)
    t0 = time.time()
    sh(f'cp -r "{DRIVE_SRC}" "{LOCAL_SRC}"')
    print(f"  staged in {time.time() - t0:.0f}s", flush=True)


def mel_from_wav(wav_path):
    import librosa

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]

    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window="hann", center=True, power=2.0,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )

    if LOG_SCALE:
        S = librosa.power_to_db(S, ref=np.max)

    if MINMAX_01:
        lo, hi = float(S.min()), float(S.max())
        S = (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)

    return S.astype(DTYPE)


def process_one(job):
    src, dst = job
    try:
        arr = mel_from_wav(src)
        np.save(dst, arr)
        return True, arr.shape
    except Exception as exc:                    
        return False, f"{src}: {exc}"


def collect_jobs():
    jobs = []
    per_split = {}
    for split in SPLITS:
        split_dir = LOCAL_SRC / split
        if not split_dir.is_dir():
            print(f"  !! split folder missing: {split_dir}", flush=True)
            continue
        counts = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            out_dir = LOCAL_OUT / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)   
            wavs = sorted(class_dir.glob("*.wav"))
            for w in wavs:
                jobs.append((w, out_dir / f"{w.stem}.npy"))
            counts[class_dir.name] = len(wavs)
        per_split[split] = counts
    return jobs, per_split


def main():
    try:
        import librosa                                  
    except ImportError:
        sys.exit("this script needs librosa:  pip install librosa soundfile")

    print(f"mel config: n_fft {N_FFT} | win {WIN_LENGTH} | hop {HOP_LENGTH} | "
          f"n_mels {N_MELS} | sr {SR}", flush=True)
    print(f"  -> {1 + CLIP_SAMPLES // HOP_LENGTH} frames per 1s clip "
          f"(arrays are {N_MELS} x {1 + CLIP_SAMPLES // HOP_LENGTH})", flush=True)
    print(f"  log_scale={LOG_SCALE}  minmax01={MINMAX_01}  dtype={np.dtype(DTYPE).name}\n",
          flush=True)

    stage_source()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    jobs, per_split = collect_jobs()
    if not jobs:
        raise SystemExit(f"no .wav files found under {LOCAL_SRC}")

    print("source clips:")
    for split, counts in per_split.items():
        total = sum(counts.values())
        detail = "  ".join(f"{c}={n}" for c, n in sorted(counts.items()))
        print(f"  {split:11} {total:>6}   {detail}", flush=True)
    print(f"  {'TOTAL':11} {len(jobs):>6}\n", flush=True)

    print(f"generating {len(jobs)} mel arrays on {WORKERS} workers...", flush=True)
    t0 = time.time()
    done, failed, shapes = 0, [], set()
    with Pool(WORKERS) as pool:
        for ok, info in pool.imap_unordered(process_one, jobs, chunksize=32):
            if ok:
                shapes.add(info)
            else:
                failed.append(info)
            done += 1
            if done % 1000 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(jobs)}  ({rate:.0f} clips/s)", flush=True)
    print(f"  finished in {time.time() - t0:.0f}s", flush=True)

    print(f"\narray shapes produced: {sorted(shapes)}", flush=True)
    if len(shapes) > 1:
        print("  !! more than one shape - check for clips that were not 1s", flush=True)
    if failed:
        print(f"  !! {len(failed)} clip(s) failed:", flush=True)
        for msg in failed[:10]:
            print(f"     {msg}", flush=True)

    n_npy = sum(1 for _ in LOCAL_OUT.rglob("*.npy"))
    size_mb = sum(p.stat().st_size for p in LOCAL_OUT.rglob("*.npy")) / 1e6
    print(f"\nwrote {n_npy} .npy files ({size_mb:.0f} MB) under {LOCAL_OUT}", flush=True)

    print(f"\ntarring -> {LOCAL_TAR} ...", flush=True)
    t0 = time.time()
    with tarfile.open(LOCAL_TAR, "w") as tar:               
        tar.add(LOCAL_OUT, arcname=LOCAL_OUT.name)
    print(f"  {LOCAL_TAR.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\ncopying tar -> {DRIVE_TAR} ...", flush=True)
    t0 = time.time()
    sh(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
    print(f"  copied in {time.time() - t0:.0f}s", flush=True)

    print("\ndone. at training time:")
    print(f'  !cp "{DRIVE_TAR}" /content/ && tar -xf /content/{LOCAL_TAR.name} -C /content/')
    print(f"  -> /content/{LOCAL_OUT.name}/<split>/<class>/*.npy")


if __name__ == "__main__":
    main()
