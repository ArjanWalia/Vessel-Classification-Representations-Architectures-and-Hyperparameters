

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/content/drive/MyDrive/vessel_classification_final")
COMBO_DIR = BASE / "Codes" / "Testing" / "MFCC + KNN"
MODEL_PATH = COMBO_DIR / "Outputs" / "knn_mfcc_best.joblib"
OUTPUT_DIR = COMBO_DIR / "Outputs"
SWEEP_LOG = OUTPUT_DIR / "sweep_results.json"

DATA_TAR = BASE / "Dataset" / "mfcc_dataset.tar"
LOCAL_DATA = Path("/content/mfcc_data")

TEST_SPLIT = "test"



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")



def find_data_root(base):
    base = Path(base)
    if not base.exists():
        return None
    for d in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / f"X_{TEST_SPLIT}.npy").is_file() and (d / f"y_{TEST_SPLIT}.npy").is_file():
            return d
    return None


def stage_dataset():
    root = find_data_root(LOCAL_DATA)
    if root is not None:
        print(f"dataset already staged at {root}", flush=True)
        return root

    if not DATA_TAR.is_file():
        raise SystemExit(f"dataset tar not found: {DATA_TAR}")
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    print(f"copying + unpacking {DATA_TAR.name} to local disk...", flush=True)
    t0 = time.time()
    sh(f'cp "{DATA_TAR}" /content/_mfcc.tar')
    sh(f'tar -xf /content/_mfcc.tar -C "{LOCAL_DATA}"')
    sh("rm -f /content/_mfcc.tar")
    print(f"  unpacked in {time.time() - t0:.0f}s", flush=True)

    root = find_data_root(LOCAL_DATA)
    if root is None:
        raise SystemExit(f"could not find X_{TEST_SPLIT}.npy under {LOCAL_DATA}")
    return root



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



def main():
    try:
        import joblib
        import sklearn                                  
    except ImportError:
        sys.exit("this script needs scikit-learn and joblib:  pip install scikit-learn joblib")

    if not MODEL_PATH.is_file():
        raise SystemExit(f"model not found: {MODEL_PATH}")

    root = stage_dataset()
    Xte = np.load(root / f"X_{TEST_SPLIT}.npy").astype(np.float32)
    yte = np.load(root / f"y_{TEST_SPLIT}.npy")

    classes = json.loads((root / "classes.json").read_text()) \
        if (root / "classes.json").is_file() else [str(i) for i in range(int(yte.max()) + 1)]

    print(f"\nloaded {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)", flush=True)
    model = joblib.load(MODEL_PATH)
    print(f"  {type(model).__name__}: k={getattr(model, 'n_neighbors', '?')}, "
          f"weights={getattr(model, 'weights', '?')}, "
          f"metric={getattr(model, 'metric', '?')}", flush=True)
    print(f"  (a KNN file stores the training set, not learned weights)", flush=True)

    val_acc = None
    if SWEEP_LOG.is_file():
        log = json.loads(SWEEP_LOG.read_text())
        best = log.get("best", {})
        val_acc = best.get("val_accuracy")
        print(f"  selected on validation: k={best.get('k')}, "
              f"val_accuracy={val_acc * 100:.2f}%" if val_acc is not None else "", flush=True)

    print(f"\nclasses: {classes}", flush=True)
    print(f"test {Xte.shape}  labels {yte.shape}", flush=True)
    print(f"feature check: mean {Xte.mean():+.4f}, std {Xte.std():.4f} "
          f"(standardised in the builder -> not re-scaled here)", flush=True)

    if hasattr(model, "classes_"):
        model_labels = set(int(c) for c in model.classes_)
        data_labels = set(int(c) for c in np.unique(yte))
        if not data_labels.issubset(model_labels):
            print(f"  !! label mismatch - model knows {sorted(model_labels)}, "
                  f"test contains {sorted(data_labels)}", flush=True)

    t0 = time.time()
    preds = model.predict(Xte)
    print(f"\npredicted {len(preds)} clips in {time.time() - t0:.1f}s", flush=True)

    cm = confusion_matrix(preds, yte, len(classes))
    per_class, summary = report_rows(cm, classes)
    text = format_report(per_class, summary)

    header = "Classification report - KNN + MFCC - test set"
    print("\n" + "=" * len(header), flush=True)
    print(header, flush=True)
    print("=" * len(header), flush=True)
    print(text, flush=True)
    print("=" * len(header), flush=True)

    if val_acc is not None:
        gap = val_acc - summary["accuracy"]
        print(f"\nvalidation {val_acc * 100:.2f}%  ->  test {summary['accuracy'] * 100:.2f}%"
              f"   (gap {gap * 100:+.2f} pts)", flush=True)
        print("  a modest drop is normal: k was chosen on validation.", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test_classification_report.txt").write_text(header + "\n\n" + text + "\n")
    (OUTPUT_DIR / "test_classification_report.tex").write_text(
        format_latex(per_class, summary) + "\n")
    save_confusion_matrix(cm, classes, OUTPUT_DIR / "test_confusion_matrix.png",
                          "KNN + MFCC \u2014 test")
    (OUTPUT_DIR / "test_results.json").write_text(json.dumps({
        "model": "knn", "representation": "mfcc",
        "n_neighbors": int(getattr(model, "n_neighbors", -1)),
        "weights": str(getattr(model, "weights", "")),
        "metric": str(getattr(model, "metric", "")),
        "classes": classes, "n_features": int(Xte.shape[1]),
        "val_accuracy": val_acc, "test_accuracy": summary["accuracy"],
        "per_class": [{k: (float(v) if k not in ("support", "name") else v)
                       for k, v in r.items()} for r in per_class],
        "macro_avg": {k: float(v) for k, v in summary["macro"].items()},
        "weighted_avg": {k: float(v) for k, v in summary["weighted"].items()},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(yte)),
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
