

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


BASE = Path("/content/drive/MyDrive/vessel_classification_final")
COMBO_DIR = BASE / "Codes" / "Testing" / "WAV + AST"
CKPT_PATH = COMBO_DIR / "Outputs" / "ast_wav_best.pt"
OUTPUT_DIR = COMBO_DIR / "Outputs"

DRIVE_SRC = BASE / "Dataset" / "cumulative_dataset"
DRIVE_TAR = BASE / "Dataset" / "cumulative_dataset.tar"     
LOCAL_SRC = Path("/content/cumulative_dataset")
FEATURE_CACHE = Path("/content/ast_features")                

MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"
TEST_SPLIT = "test"

BATCH = 64
NUM_WORKERS = 2
EXTRACT_THREADS = 8

SR = 16000
MAX_LENGTH = 128
NUM_MEL_BINS = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")



_FE = None
_FE_LOCK = threading.Lock()


def feature_extractor(max_length, num_mel_bins):
    global _FE
    if _FE is None:
        with _FE_LOCK:
            if _FE is None:
                _FE = ASTFeatureExtractor.from_pretrained(
                    MODEL_NAME, max_length=max_length, num_mel_bins=num_mel_bins)
    return _FE


def stage_wavs():
    if LOCAL_SRC.is_dir() and any(LOCAL_SRC.rglob("*.wav")):
        print(f"wavs already staged at {LOCAL_SRC}", flush=True)
        return
    t0 = time.time()
    if DRIVE_TAR.is_file():
        LOCAL_SRC.mkdir(parents=True, exist_ok=True)
        print(f"unpacking {DRIVE_TAR.name} to local disk...", flush=True)
        sh(f'cp "{DRIVE_TAR}" /content/_cum.tar')
        sh(f'tar -xf /content/_cum.tar -C "{LOCAL_SRC}"')
        sh("rm -f /content/_cum.tar")
    elif DRIVE_SRC.is_dir():
        print(f"copying {DRIVE_SRC} -> {LOCAL_SRC} (one bulk copy)...", flush=True)
        sh(f'cp -r "{DRIVE_SRC}" "{LOCAL_SRC}"')
    else:
        raise SystemExit(f"neither {DRIVE_TAR} nor {DRIVE_SRC} exists")
    print(f"  staged in {time.time() - t0:.0f}s", flush=True)


def find_dir_with(base, ext, split):
    base = Path(base)
    if not base.exists():
        return None
    for d in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / split).is_dir() and next((d / split).rglob(f"*{ext}"), None) is not None:
            return d
    return None


def build_test_features(classes, sr, max_length, num_mel_bins):
    cached = find_dir_with(FEATURE_CACHE, ".npy", TEST_SPLIT)
    if cached is not None:
        print(f"reusing feature cache at {cached / TEST_SPLIT}", flush=True)
        root = cached
    else:
        print("no feature cache found - extracting the test split", flush=True)
        stage_wavs()
        wav_root = find_dir_with(LOCAL_SRC, ".wav", TEST_SPLIT)
        if wav_root is None:
            raise SystemExit(f"no '{TEST_SPLIT}' wav split under {LOCAL_SRC}")

        fe = feature_extractor(max_length, num_mel_bins)
        jobs = []
        for cls in classes:
            cdir = wav_root / TEST_SPLIT / cls
            out_dir = FEATURE_CACHE / TEST_SPLIT / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            if not cdir.is_dir():
                print(f"  !! {TEST_SPLIT}: no folder for class '{cls}'", flush=True)
                continue
            for w in sorted(cdir.glob("*.wav")):
                jobs.append((w, out_dir / f"{w.stem}.npy"))

        def one(job):
            src, dst = job
            if dst.exists() and dst.stat().st_size > 0:
                return
            import librosa
            y, _ = librosa.load(str(src), sr=sr, mono=True)
            y = np.pad(y, (0, sr - len(y))) if len(y) < sr else y[:sr]
            arr = fe(y, sampling_rate=sr, return_tensors="np")["input_values"][0]
            np.save(dst, arr.astype(np.float32))

        print(f"  extracting {len(jobs)} clips on {EXTRACT_THREADS} threads...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=EXTRACT_THREADS) as pool:
            list(tqdm(pool.map(one, jobs), total=len(jobs), desc="extract"))
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
        root = FEATURE_CACHE

    samples = []
    for idx, cls in enumerate(classes):
        cdir = root / TEST_SPLIT / cls
        if not cdir.is_dir():
            print(f"  !! {TEST_SPLIT}: no folder for class '{cls}'", flush=True)
            continue
        for npy in sorted(cdir.glob("*.npy")):
            samples.append((npy, idx))
    return samples


class FeatureDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return torch.from_numpy(np.load(path).astype(np.float32)), label



def confusion_matrix(preds, labels, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def report_rows(cm, classes):
    support = cm.sum(axis=1).astype(float)
    pred_tot = cm.sum(axis=0).astype(float)
    tp = np.diag(cm).astype(float)

    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    total = support.sum()
    accuracy = tp.sum() / total if total > 0 else 0.0
    w = support / total if total > 0 else np.zeros_like(support)

    per_class = [{"name": c, "precision": precision[i], "recall": recall[i],
                  "f1": f1[i], "support": int(support[i])}
                 for i, c in enumerate(classes)]
    summary = {
        "accuracy": accuracy,
        "macro": {"precision": precision.mean(), "recall": recall.mean(),
                  "f1": f1.mean(), "support": int(total)},
        "weighted": {"precision": float((precision * w).sum()),
                     "recall": float((recall * w).sum()),
                     "f1": float((f1 * w).sum()), "support": int(total)},
    }
    return per_class, summary


def format_report(per_class, summary):
    name_w = max(14, max(len(r["name"]) for r in per_class) + 2)
    out = [f"{'':<{name_w}}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for r in per_class:
        out.append(f"{r['name']:<{name_w}}{r['precision']:>10.4f}{r['recall']:>10.4f}"
                   f"{r['f1']:>10.4f}{r['support']:>10d}")
    out.append("")
    s = summary
    out.append(f"{'accuracy':<{name_w}}{'':>10}{'':>10}{s['accuracy']:>10.4f}"
               f"{s['macro']['support']:>10d}")
    for key, label in (("macro", "macro avg"), ("weighted", "weighted avg")):
        m = s[key]
        out.append(f"{label:<{name_w}}{m['precision']:>10.4f}{m['recall']:>10.4f}"
                   f"{m['f1']:>10.4f}{m['support']:>10d}")
    return "\n".join(out)


def format_latex(per_class, summary):
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Classification Report}",
         r"\label{tab:class_report}", r"\begin{tabular}{lcccc}", r"\toprule",
         r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & "
         r"\textbf{F1-score} & \textbf{Support} \\ \midrule"]
    for r in per_class:
        L.append(f"{r['name']:<14} & {r['precision']:.2f} & {r['recall']:.2f} & "
                 f"{r['f1']:.2f} & {r['support']} \\\\")
    L.append(r"\midrule")
    s = summary
    L.append(f"accuracy       &      &      & {s['accuracy']:.2f} & "
             f"{s['macro']['support']} \\\\")
    for key, label in (("macro", "macro avg"), ("weighted", "weighted avg")):
        m = s[key]
        L.append(f"{label:<14} & {m['precision']:.2f} & {m['recall']:.2f} & "
                 f"{m['f1']:.2f} & {m['support']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def save_confusion_matrix(cm, classes, out_png, title):
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row, 1), dtype=float)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=45, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center",
                    va="center", fontsize=9,
                    color=("white" if norm[i, j] > 0.5 else "black"))
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)



@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in tqdm(loader, desc="test"):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = model(input_values=x).logits
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main():
    try:
        import librosa                                   
    except ImportError:
        sys.exit("this script needs librosa:  pip install librosa")

    print(f"Device: {DEVICE}", flush=True)
    if not CKPT_PATH.is_file():
        raise SystemExit(f"checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    state = ckpt["model_state"]
    hp = ckpt.get("hyperparameters", {})
    fx = ckpt.get("feature_extraction", {})
    sr = fx.get("sampling_rate", SR)
    max_length = fx.get("max_length", MAX_LENGTH)
    num_mel_bins = fx.get("num_mel_bins", NUM_MEL_BINS)

    print(f"\nloaded {CKPT_PATH.name}", flush=True)
    print(f"  model {ckpt.get('model')} ({ckpt.get('model_name')})", flush=True)
    print(f"  representation {ckpt.get('representation')}", flush=True)
    print(f"  classes {classes}", flush=True)
    print(f"  config: lr {hp.get('lr')}, epochs {hp.get('epochs')}, "
          f"batch {hp.get('batch')}, {hp.get('optimizer')}", flush=True)
    print(f"  features: {max_length} x {num_mel_bins} @ {sr} Hz  "
          f"(read from the checkpoint)", flush=True)
    if "val_accuracy" in ckpt:
        print(f"  validation accuracy at selection: "
              f"{ckpt['val_accuracy'] * 100:.2f}%", flush=True)
    print("", flush=True)

    samples = build_test_features(classes, sr, max_length, num_mel_bins)
    if not samples:
        raise SystemExit("no test features found")
    ds = FeatureDataset(samples)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    probe, _ = ds[0]
    print(f"\ntest clips {len(ds)}  |  input shape {tuple(probe.shape)}  "
          f"(expect ({max_length}, {num_mel_bins}))\n", flush=True)

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes), ignore_mismatched_sizes=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)

    preds, labels = predict(model, loader)
    cm = confusion_matrix(preds, labels, len(classes))
    per_class, summary = report_rows(cm, classes)
    text = format_report(per_class, summary)

    header = "Classification report - AST + raw WAV - test set"
    print("\n" + "=" * len(header), flush=True)
    print(header, flush=True)
    print("=" * len(header), flush=True)
    print(text, flush=True)
    print("=" * len(header), flush=True)

    if "val_accuracy" in ckpt:
        gap = ckpt["val_accuracy"] - summary["accuracy"]
        print(f"\nvalidation {ckpt['val_accuracy'] * 100:.2f}%  ->  "
              f"test {summary['accuracy'] * 100:.2f}%   (gap {gap * 100:+.2f} pts)",
              flush=True)
        print("  a modest drop is normal: the configuration was chosen on validation.",
              flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test_classification_report.txt").write_text(header + "\n\n" + text + "\n")
    (OUTPUT_DIR / "test_classification_report.tex").write_text(
        format_latex(per_class, summary) + "\n")
    save_confusion_matrix(cm, classes, OUTPUT_DIR / "test_confusion_matrix.png",
                          "AST + raw WAV \u2014 test")
    (OUTPUT_DIR / "test_results.json").write_text(json.dumps({
        "model": ckpt.get("model"), "model_name": ckpt.get("model_name"),
        "representation": ckpt.get("representation"), "classes": classes,
        "hyperparameters": hp,
        "feature_extraction": {"sampling_rate": sr, "max_length": max_length,
                               "num_mel_bins": num_mel_bins},
        "val_accuracy": ckpt.get("val_accuracy"),
        "test_accuracy": summary["accuracy"],
        "per_class": [{k: (float(v) if k not in ("support", "name") else v)
                       for k, v in r.items()} for r in per_class],
        "macro_avg": {k: float(v) for k, v in summary["macro"].items()},
        "weighted_avg": {k: float(v) for k, v in summary["weighted"].items()},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(ds)),
    }, indent=2))

    print(f"\nsaved to {OUTPUT_DIR}:", flush=True)
    for f in ("test_classification_report.txt", "test_classification_report.tex",
              "test_confusion_matrix.png", "test_results.json"):
        print(f"  {f}", flush=True)

    w = summary["weighted"]
    print(f"\nTable 3 row:  accuracy {summary['accuracy'] * 100:.2f}  "
          f"precision {w['precision'] * 100:.2f}  recall {w['recall'] * 100:.2f}  "
          f"F1 {w['f1'] * 100:.2f}", flush=True)


if __name__ == "__main__":
    main()
