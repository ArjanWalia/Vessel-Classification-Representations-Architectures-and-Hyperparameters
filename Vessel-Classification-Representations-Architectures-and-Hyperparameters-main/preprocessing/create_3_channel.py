import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

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
CLIP_SAMPLES = SR                
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH       


N_MELS = 128
MEL_FMIN, MEL_FMAX = 0, SR // 2

CQT_N_BINS = 128
CQT_BINS_PER_OCTAVE = 16
CQT_FMIN = 30.0

N_GAMMA = 128
GAMMA_FMIN, GAMMA_FMAX = 30.0, SR // 2
GAMMA_ORDER = 4

TOP_DB = 80.0                   
DTYPE = np.float32                

WORKERS = max(1, cpu_count() - 1)
STAGE_FROM_DRIVE = True



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")



def _erb(f):
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def _erb_space(fmin, fmax, n):
    ear_q, min_bw = 9.26449, 24.7
    idx = np.arange(1, n + 1)
    cf = -(ear_q * min_bw) + np.exp(
        idx * (-np.log(fmax + ear_q * min_bw) + np.log(fmin + ear_q * min_bw)) / n
    ) * (fmax + ear_q * min_bw)
    return cf[::-1]                                    


def gammatone_weights():
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    cf = _erb_space(GAMMA_FMIN, GAMMA_FMAX, N_GAMMA)
    bw = 1.019 * _erb(cf)
    w = (1.0 + ((freqs[None, :] - cf[:, None]) / bw[:, None]) ** 2) ** (-GAMMA_ORDER / 2.0)
    w /= np.maximum(w.max(axis=1, keepdims=True), 1e-10)   
    return w


_GAMMA_W = None          


def _gamma_w():
    global _GAMMA_W
    if _GAMMA_W is None:
        _GAMMA_W = gammatone_weights()
    return _GAMMA_W



def to_unit_db(power, ref_max=True):
    import librosa
    db = librosa.power_to_db(np.maximum(power, 1e-10),
                             ref=np.max if ref_max else 1.0, top_db=TOP_DB)
    lo, hi = float(db.min()), float(db.max())
    if hi <= lo:                                     
        return np.zeros_like(db)
    return (db - lo) / (hi - lo)


def mcg_from_wav(wav_path):
    import librosa

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann", center=True)
    power = np.abs(stft) ** 2                          

    mel = librosa.feature.melspectrogram(S=power, sr=SR, n_mels=N_MELS,
                                         fmin=MEL_FMIN, fmax=MEL_FMAX)
    gam = _gamma_w() @ power                         

    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=CQT_FMIN,
                             n_bins=CQT_N_BINS,
                             bins_per_octave=CQT_BINS_PER_OCTAVE)) ** 2

    t = min(mel.shape[1], cqt.shape[1], gam.shape[1])
    mel, cqt, gam = mel[:, :t], cqt[:, :t], gam[:, :t]

    chans = [to_unit_db(mel), to_unit_db(cqt), to_unit_db(gam)]
    return np.stack(chans, axis=0).astype(DTYPE)     


def process_one(job):
    src, dst = job
    try:
        arr = mcg_from_wav(src)
        np.save(dst, arr)
        return True, arr.shape
    except Exception as exc:                         
        return False, f"{src}: {exc}"


def stage_source():
    if LOCAL_SRC.is_dir() and any(LOCAL_SRC.rglob("*.wav")):
        print(f"source already staged at {LOCAL_SRC}", flush=True)
        return
    if not STAGE_FROM_DRIVE:
        raise SystemExit(f"{LOCAL_SRC} missing and STAGE_FROM_DRIVE is False")
    if not DRIVE_SRC.is_dir():
        raise SystemExit(f"source not found on Drive: {DRIVE_SRC}")
    print(f"staging {DRIVE_SRC} -> {LOCAL_SRC} (one bulk copy)...", flush=True)
    t0 = time.time()
    sh(f'cp -r "{DRIVE_SRC}" "{LOCAL_SRC}"')
    print(f"  staged in {time.time() - t0:.0f}s", flush=True)


def collect_jobs():
    jobs, per_split = [], {}
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

    print(f"STFT: n_fft {N_FFT} | win {WIN_LENGTH} | hop {HOP_LENGTH} | sr {SR}", flush=True)
    print(f"  ch0 mel   : {N_MELS} bands, {MEL_FMIN}-{MEL_FMAX} Hz", flush=True)
    print(f"  ch1 cqt   : {CQT_N_BINS} bins, {CQT_BINS_PER_OCTAVE}/octave from "
          f"{CQT_FMIN:.0f} Hz -> top {CQT_FMIN * 2 ** (CQT_N_BINS / CQT_BINS_PER_OCTAVE):.0f} Hz",
          flush=True)
    print(f"  ch2 gamma : {N_GAMMA} ERB filters, {GAMMA_FMIN:.0f}-{GAMMA_FMAX} Hz", flush=True)
    print(f"  -> arrays are (3, {N_MELS}, {N_FRAMES}) CHW, {np.dtype(DTYPE).name}, "
          f"each channel dB + min-max to [0,1]\n", flush=True)

    stage_source()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    jobs, per_split = collect_jobs()
    if not jobs:
        raise SystemExit(f"no .wav files found under {LOCAL_SRC}")

    print("source clips:")
    for split, counts in per_split.items():
        detail = "  ".join(f"{c}={n}" for c, n in sorted(counts.items()))
        print(f"  {split:11} {sum(counts.values()):>6}   {detail}", flush=True)
    print(f"  {'TOTAL':11} {len(jobs):>6}\n", flush=True)

    est_gb = len(jobs) * 3 * N_MELS * N_FRAMES * np.dtype(DTYPE).itemsize / 1e9
    print(f"generating {len(jobs)} arrays on {WORKERS} workers "
          f"(~{est_gb:.2f} GB expected; CQT is the slow channel)...", flush=True)
    t0 = time.time()
    done, failed, shapes = 0, [], set()
    with Pool(WORKERS) as pool:
        for ok, info in pool.imap_unordered(process_one, jobs, chunksize=16):
            if ok:
                shapes.add(info)
            else:
                failed.append(info)
            done += 1
            if done % 500 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(jobs)}  ({rate:.1f} clips/s, eta {eta/60:.1f} min)",
                      flush=True)
    print(f"  finished in {(time.time() - t0)/60:.1f} min", flush=True)

    print(f"\narray shapes produced: {sorted(shapes)}", flush=True)
    if len(shapes) > 1:
        print("  !! more than one shape - inspect before training", flush=True)
    if failed:
        print(f"  !! {len(failed)} clip(s) failed:", flush=True)
        for msg in failed[:10]:
            print(f"     {msg}", flush=True)

    n_npy = sum(1 for _ in LOCAL_OUT.rglob("*.npy"))
    size_gb = sum(p.stat().st_size for p in LOCAL_OUT.rglob("*.npy")) / 1e9
    print(f"\nwrote {n_npy} .npy files ({size_gb:.2f} GB) under {LOCAL_OUT}", flush=True)

    print(f"\ntarring -> {LOCAL_TAR} ...", flush=True)
    t0 = time.time()
    with tarfile.open(LOCAL_TAR, "w") as tar:           
        tar.add(LOCAL_OUT, arcname=LOCAL_OUT.name)
    print(f"  {LOCAL_TAR.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\ncopying tar -> {DRIVE_TAR} ...", flush=True)
    t0 = time.time()
    sh(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
    print(f"  copied in {time.time() - t0:.0f}s", flush=True)

    print("\ndone. at training time:")
    print(f'  !cp "{DRIVE_TAR}" /content/ && tar -xf /content/{LOCAL_TAR.name} -C /content/')
    print(f"  -> /content/{LOCAL_OUT.name}/<split>/<class>/*.npy   (3, {N_MELS}, {N_FRAMES}) CHW")


if __name__ == "__main__":
    main()
