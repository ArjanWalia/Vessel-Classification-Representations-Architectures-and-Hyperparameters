

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


BASE = Path("/content/drive/MyDrive/vessel_classification_final")
DATA_TAR = BASE / "Dataset" / "mfcc_dataset.tar"
GSHEET_PATH = BASE / "Documentation" / "Testing.gsheet"
OUTPUT_DIR = BASE / "Codes" / "Testing" / "MFCC + KNN" / "Outputs"

SHEET_TAB = "MFCC + KNN"
GSHEET_TITLE_FALLBACK = "Testing"
CREDENTIALS_JSON = "credentials.json"

LOCAL_DATA = Path("/content/mfcc_data")


NEIGHBORS = list(range(1, 15))

WEIGHTS = "distance"
METRIC = "minkowski"
P_NORM = 2
ALGORITHM = "auto"
N_JOBS = -1



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")



def find_data_root(base):
    base = Path(base)
    if not base.exists():
        return None
    for d in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / "X_train.npy").is_file() and (d / "X_validation.npy").is_file():
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
        raise SystemExit(f"could not find X_train.npy / X_validation.npy under {LOCAL_DATA}")
    return root


def load_split(root, split):
    X = np.load(root / f"X_{split}.npy").astype(np.float32)
    y = np.load(root / f"y_{split}.npy")
    return X, y



def open_sheet():
    import gspread
    gc = None
    if Path(CREDENTIALS_JSON).is_file():
        try:
            gc = gspread.service_account(filename=CREDENTIALS_JSON)
        except Exception as exc:
            print(f"  service_account auth failed ({exc}); trying Colab auth", flush=True)
    if gc is None:
        from google.colab import auth
        auth.authenticate_user()
        from google.auth import default
        creds, _ = default()
        gc = gspread.authorize(creds)

    doc_id = None
    try:
        meta = json.loads(Path(GSHEET_PATH).read_text())
        doc_id = meta.get("doc_id") or re.search(r"/d/([A-Za-z0-9_-]+)",
                                                 meta.get("url", "")).group(1)
    except Exception:
        pass
    return gc.open_by_key(doc_id) if doc_id else gc.open(GSHEET_TITLE_FALLBACK)


def _update(ws, rng, values):
    try:
        ws.update(range_name=rng, values=values, value_input_option="USER_ENTERED")
    except TypeError:
        ws.update(rng, values, value_input_option="USER_ENTERED")


def get_ws(spreadsheet):
    try:
        return spreadsheet.worksheet(SHEET_TAB)
    except Exception:
        ws = spreadsheet.add_worksheet(title=SHEET_TAB,
                                       rows=len(NEIGHBORS) + 2, cols=4)
        _update(ws, "A1:B1", [["Neighbors", "Accuracy"]])
        _update(ws, f"A2:A{len(NEIGHBORS) + 1}", [[k] for k in NEIGHBORS])
        return ws


def write_accuracy(ws, row_index, value):
    cell = f"B{row_index + 2}"
    _update(ws, f"{cell}:{cell}", [[value]])



def main():
    try:
        from sklearn.neighbors import KNeighborsClassifier
        import joblib
    except ImportError:
        sys.exit("this script needs scikit-learn and joblib:  pip install scikit-learn joblib")

    print("K-Nearest Neighbors + MFCC (mean+std aggregated)", flush=True)
    print(f"weights {WEIGHTS} | metric {METRIC} (p={P_NORM}) | algorithm {ALGORITHM}",
          flush=True)
    print(f"neighbor sweep: k = {NEIGHBORS[0]}..{NEIGHBORS[-1]}\n", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spreadsheet = open_sheet()
    ws = get_ws(spreadsheet)
    print(f"sheet '{spreadsheet.title}' tab '{SHEET_TAB}'\n", flush=True)

    root = stage_dataset()
    Xtr, ytr = load_split(root, "train")
    Xva, yva = load_split(root, "validation")
    classes = json.loads((root / "classes.json").read_text()) \
        if (root / "classes.json").is_file() else None

    print(f"train {Xtr.shape}  validation {Xva.shape}", flush=True)
    print(f"classes: {classes}", flush=True)
    print(f"feature check: train mean {Xtr.mean():+.4f}, std {Xtr.std():.4f} "
          f"(already standardized -> not re-scaled here)\n", flush=True)

    best = {"acc": -1.0, "k": None}
    results = []
    t_all = time.time()

    for i, k in enumerate(NEIGHBORS):
        t0 = time.time()
        model = KNeighborsClassifier(
            n_neighbors=k, weights=WEIGHTS, metric=METRIC, p=P_NORM,
            algorithm=ALGORITHM, n_jobs=N_JOBS)
        model.fit(Xtr, ytr)
        acc = float(model.score(Xva, yva))
        secs = time.time() - t0

        write_accuracy(ws, i, round(acc, 4))
        parity = "even" if k % 2 == 0 else "odd"
        print(f"  k {k:>3} ({parity:<4}) -> val accuracy {acc * 100:6.2f}%   "
              f"({secs:.1f}s)   -> B{i + 2}", flush=True)

        results.append({"k": k, "accuracy": acc, "seconds": round(secs, 2)})

        if acc > best["acc"]:
            best = {"acc": acc, "k": k}
            joblib.dump(model, OUTPUT_DIR / "knn_mfcc_best.joblib")
            print(f"    >> NEW BEST (k={k})", flush=True)

    accs = [r["accuracy"] for r in results]
    print("\n" + "=" * 52, flush=True)
    print(f"{'k':>4}{'val acc':>11}{'parity':>9}", flush=True)
    for r in results:
        mark = "  <- best" if r["k"] == best["k"] else ""
        print(f"{r['k']:>4}{r['accuracy'] * 100:>10.2f}%"
              f"{('even' if r['k'] % 2 == 0 else 'odd'):>9}{mark}", flush=True)
    print("=" * 52, flush=True)

    odd = [r["accuracy"] for r in results if r["k"] % 2 == 1]
    even = [r["accuracy"] for r in results if r["k"] % 2 == 0]
    if odd and even:
        print(f"\nmean accuracy  odd k {np.mean(odd) * 100:.2f}%  |  "
              f"even k {np.mean(even) * 100:.2f}%", flush=True)
        if np.mean(odd) > np.mean(even):
            print("  odd k scoring higher is consistent with tie-breaking losses at even k;", flush=True)
            print("  WEIGHTS='distance' would remove ties if you want to test that.", flush=True)

    if best["k"] == NEIGHBORS[-1]:
        print(f"\nnote: the best k is the largest tested ({NEIGHBORS[-1]}) - the curve may", flush=True)
        print("still be rising, so extending the sweep could find a better value.", flush=True)
    if best["k"] == 1:
        print("\nnote: k=1 winning usually means the classes are tightly clustered, but it", flush=True)
        print("is also the most overfit-prone setting - check it holds up on test.", flush=True)

    print(f"\nBEST: k {best['k']}, val accuracy {best['acc'] * 100:.2f}%", flush=True)
    print(f"  model -> {OUTPUT_DIR / 'knn_mfcc_best.joblib'}", flush=True)
    print(f"  sweep completed in {time.time() - t_all:.0f}s", flush=True)

    (OUTPUT_DIR / "sweep_results.json").write_text(json.dumps({
        "model": "knn", "representation": "mfcc",
        "weights": WEIGHTS, "metric": METRIC, "p": P_NORM, "algorithm": ALGORITHM,
        "classes": classes, "n_features": int(Xtr.shape[1]),
        "n_train": int(Xtr.shape[0]), "n_validation": int(Xva.shape[0]),
        "best": {"k": best["k"], "val_accuracy": best["acc"]},
        "results": results,
    }, indent=2))
    print(f"  sweep log -> {OUTPUT_DIR / 'sweep_results.json'}", flush=True)
    print("\nTHE TASK HAS BEEN COMPLETED.", flush=True)


if __name__ == "__main__":
    main()
