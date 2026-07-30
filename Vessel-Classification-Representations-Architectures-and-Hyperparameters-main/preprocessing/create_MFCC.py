

import json
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np


DRIVE_ROOT = Path("")
DRIVE_SRC = DRIVE_ROOT / ""
DRIVE_SRC_TAR = DRIVE_ROOT / ""
DRIVE_TAR = DRIVE_ROOT / ""

LOCAL_SRC = Path("")
LOCAL_OUT = Path("")
LOCAL_TAR = Path("")

SPLITS = ["train", "validation", "test"]
FIT_SPLIT = "train"

SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 128
CLIP_SAMPLES = SR

N_MELS = 128
FMIN, FMAX = 0, 8000

N_MFCC = 20
DCT_TYPE = 2
DCT_NORM = "ortho"
LIFTER = 0
DROP_C0 = False

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

    t0 = time.time()
    if DRIVE_SRC_TAR.is_file():
        LOCAL_SRC.mkdir(parents=True, exist_ok=True)
        print(f"unpacking {DRIVE_SRC_TAR.name} to local disk...", flush=True)
        sh(f'cp "{DRIVE_SRC_TAR}" /content/_cum.tar')
        sh(f'tar -xf /content/_cum.tar -C "{LOCAL_SRC}"')
        sh("rm -f /content/_cum.tar")
    elif DRIVE_SRC.is_dir():
        print(f"copying {DRIVE_SRC} -> {LOCAL_SRC} (one bulk copy)...", flush=True)
        sh(f'cp -r "{DRIVE_SRC}" "{LOCAL_SRC}"')
    else:
        raise SystemExit(f"neither {DRIVE_SRC_TAR} nor {DRIVE_SRC} exists")
    print(f"  staged in {time.time() - t0:.0f}s", flush=True)


def find_split_root(base):
    base = Path(base)
    if not base.exists():
        return None
    for d in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / "train").is_dir() and (d / "validation").is_dir() \
                and next((d / "validation").rglob("*.wav"), None) is not None:
            return d
    return None


def mfcc_features(wav_path):
    import librosa

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]

    m = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, dct_type=DCT_TYPE, norm=DCT_NORM, lifter=LIFTER,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window="hann", center=True, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )

    if DROP_C0:
        m = m[1:]
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def process_one(job):
    path, label = job
    try:
        return True, (mfcc_features(path), label, path.name)
    except Exception as exc:
        return False, f"{path}: {exc}"


def collect_jobs(split_dir, class_to_idx):
    jobs, counts = [], {}
    for cls, idx in class_to_idx.items():
        cdir = split_dir / cls
        wavs = sorted(cdir.glob("*.wav")) if cdir.is_dir() else []
        if not cdir.is_dir():
            print(f"  !! {split_dir.name}: no folder for class '{cls}'", flush=True)
        for w in wavs:
            jobs.append((w, idx))
        counts[cls] = len(wavs)
    return jobs, counts


def feature_names():
    idx = range(1, N_MFCC) if DROP_C0 else range(N_MFCC)
    return [f"mfcc{i}_mean" for i in idx] + [f"mfcc{i}_std" for i in idx]


def main():
    try:
        import librosa
    except ImportError:
        sys.exit("this script needs librosa:  pip install librosa soundfile")

    n_feat = 2 * (N_MFCC - 1 if DROP_C0 else N_MFCC)
    print(f"MFCC: n_mfcc {N_MFCC} (C0 {'dropped' if DROP_C0 else 'kept'}) | "
          f"dct {DCT_TYPE}/{DCT_NORM} | lifter {LIFTER}", flush=True)
    print(f"STFT: n_fft {N_FFT} | win {WIN_LENGTH} | hop {HOP_LENGTH} | sr {SR}", flush=True)
    print(f"filterbank: {N_MELS} mels, {FMIN}-{FMAX} Hz", flush=True)
    print(f"aggregation: mean + std over time -> {n_feat} features per clip\n", flush=True)

    stage_source()
    root = find_split_root(LOCAL_SRC)
    if root is None:
        raise SystemExit(f"could not find train/validation wav splits under {LOCAL_SRC}")
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"classes: {classes}\n", flush=True)

    raw, labels, names = {}, {}, {}
    for split in SPLITS:
        sdir = root / split
        if not sdir.is_dir():
            print(f"  !! split folder missing: {sdir}", flush=True)
            continue
        jobs, counts = collect_jobs(sdir, class_to_idx)
        if not jobs:
            print(f"  !! no wavs in {sdir}", flush=True)
            continue

        detail = "  ".join(f"{c}={n}" for c, n in sorted(counts.items()))
        print(f"[{split}] {len(jobs)} clips   {detail}", flush=True)

        t0, feats, labs, fnames, failed = time.time(), [], [], [], []
        with Pool(WORKERS) as pool:
            for ok, info in pool.imap(process_one, jobs, chunksize=64):
                if ok:
                    v, lab, nm = info
                    feats.append(v); labs.append(lab); fnames.append(nm)
                else:
                    failed.append(info)

        raw[split] = np.stack(feats).astype(np.float32)
        labels[split] = np.asarray(labs, dtype=np.int64)
        names[split] = fnames
        print(f"    -> {raw[split].shape} in {time.time() - t0:.0f}s", flush=True)
        if failed:
            print(f"    !! {len(failed)} clip(s) failed:", flush=True)
            for msg in failed[:5]:
                print(f"       {msg}", flush=True)

    if FIT_SPLIT not in raw:
        raise SystemExit(f"'{FIT_SPLIT}' split produced no features; cannot fit scaler")

    mean = raw[FIT_SPLIT].mean(axis=0)
    scale = raw[FIT_SPLIT].std(axis=0)
    scale[scale < 1e-8] = 1.0                            # guard constant features
    print(f"\nscaler fit on '{FIT_SPLIT}' ({raw[FIT_SPLIT].shape[0]} clips) "
          f"and applied to {', '.join(raw)}", flush=True)

    for split in raw:
        X = ((raw[split] - mean) / scale).astype(np.float32)
        np.save(LOCAL_OUT / f"X_{split}.npy", X)
        np.save(LOCAL_OUT / f"X_{split}_raw.npy", raw[split])
        np.save(LOCAL_OUT / f"y_{split}.npy", labels[split])
        (LOCAL_OUT / f"files_{split}.json").write_text(json.dumps(names[split]))
        print(f"  {split:11} X {X.shape}  mean {X.mean():+.3f}  std {X.std():.3f}",
              flush=True)

    np.save(LOCAL_OUT / "scaler_mean.npy", mean.astype(np.float32))
    np.save(LOCAL_OUT / "scaler_scale.npy", scale.astype(np.float32))
    (LOCAL_OUT / "classes.json").write_text(json.dumps(classes))
    (LOCAL_OUT / "metadata.json").write_text(json.dumps({
        "sr": SR, "n_fft": N_FFT, "win_length": WIN_LENGTH, "hop_length": HOP_LENGTH,
        "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
        "n_mfcc": N_MFCC, "dct_type": DCT_TYPE, "norm": DCT_NORM, "lifter": LIFTER,
        "drop_c0": DROP_C0, "aggregation": "mean+std over time",
        "n_features": n_feat, "feature_names": feature_names(),
        "classes": classes, "scaler_fit_split": FIT_SPLIT,
        "counts": {s: int(raw[s].shape[0]) for s in raw},
    }, indent=2))

    print(f"\ntarring -> {LOCAL_TAR} ...", flush=True)
    with tarfile.open(LOCAL_TAR, "w") as tar:
        tar.add(LOCAL_OUT, arcname=LOCAL_OUT.name)
    print(f"  {LOCAL_TAR.stat().st_size / 1e6:.2f} MB", flush=True)

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    sh(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
    print(f"copied -> {DRIVE_TAR}", flush=True)

    print("\nto use:")
    print(f'  !cp "{DRIVE_TAR}" /content/ && tar -xf /content/{LOCAL_TAR.name} -C /content/')
    print("  X = np.load('/content/mfcc_dataset/X_train.npy')   # already standardized")
    print("  y = np.load('/content/mfcc_dataset/y_train.npy')")


if __name__ == "__main__":
    main()
