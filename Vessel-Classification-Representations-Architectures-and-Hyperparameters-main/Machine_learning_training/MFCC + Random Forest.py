

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
OUTPUT_DIR = BASE / "Codes" / "Testing" / "MFCC + Random Forest" / "Outputs"

SHEET_TAB = "MFCC + Random Forest"
GSHEET_TITLE_FALLBACK = "Testing"
CREDENTIALS_JSON = "credentials.json"

LOCAL_DATA = Path("/content/mfcc_data")


SHEET_ESTIMATORS = [1, 2, 3, 4, 5, 6, 7]
SHEET_MAX_DEPTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

CRITERION = "gini"
MAX_FEATURES = "sqrt"
RANDOM_STATE = 42
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


def get_grid_ws(spreadsheet):
    from gspread.utils import rowcol_to_a1
    try:
        return spreadsheet.worksheet(SHEET_TAB)
    except Exception:
        ws = spreadsheet.add_worksheet(title=SHEET_TAB,
                                       rows=max(10, len(SHEET_ESTIMATORS) + 2),
                                       cols=max(12, len(SHEET_MAX_DEPTHS) + 2))
        hdr_end = rowcol_to_a1(1, 1 + len(SHEET_MAX_DEPTHS))
        _update(ws, f"A1:{hdr_end}", [["estimators/max_depth"] + SHEET_MAX_DEPTHS])
        lab_end = rowcol_to_a1(1 + len(SHEET_ESTIMATORS), 1)
        _update(ws, f"A2:{lab_end}", [[n] for n in SHEET_ESTIMATORS])
        return ws


def write_cell(ws, est_index, depth_index, value):
    from gspread.utils import rowcol_to_a1
    cell = rowcol_to_a1(est_index + 2, depth_index + 2)
    _update(ws, f"{cell}:{cell}", [[value]])



def main():
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    except ImportError:
        sys.exit("this script needs scikit-learn and joblib:  pip install scikit-learn joblib")

    print("Random Forest + MFCC (mean+std aggregated)", flush=True)
    print(f"criterion {CRITERION} | max_features {MAX_FEATURES} | "
          f"random_state {RANDOM_STATE}", flush=True)
    print(f"grid: n_estimators {SHEET_ESTIMATORS} x max_depth {SHEET_MAX_DEPTHS} = "
          f"{len(SHEET_ESTIMATORS) * len(SHEET_MAX_DEPTHS)} cells\n", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spreadsheet = open_sheet()
    ws = get_grid_ws(spreadsheet)
    print(f"sheet '{spreadsheet.title}' tab '{SHEET_TAB}'\n", flush=True)

    root = stage_dataset()
    Xtr, ytr = load_split(root, "train")
    Xva, yva = load_split(root, "validation")
    classes = json.loads((root / "classes.json").read_text()) \
        if (root / "classes.json").is_file() else None

    print(f"train {Xtr.shape}  validation {Xva.shape}", flush=True)
    print(f"classes: {classes}\n", flush=True)

    probe = RandomForestClassifier(n_estimators=10, max_depth=None,
                                   random_state=RANDOM_STATE, n_jobs=N_JOBS).fit(Xtr, ytr)
    depths = [t.get_depth() for t in probe.estimators_]
    sat = int(np.max(depths))
    print(f"unconstrained tree depth on this data: min {min(depths)}, "
          f"mean {np.mean(depths):.1f}, max {sat}", flush=True)
    print(f"  -> max_depth >= {sat} is equivalent to unlimited; those columns will "
          f"repeat\n", flush=True)
    del probe

    best = {"acc": -1.0, "n_estimators": None, "max_depth": None}
    results = []
    grid = np.full((len(SHEET_ESTIMATORS), len(SHEET_MAX_DEPTHS)), np.nan)
    grid_t0 = time.time()

    for i, n_est in enumerate(SHEET_ESTIMATORS):
        for j, max_depth in enumerate(SHEET_MAX_DEPTHS):
            t0 = time.time()
            model = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth, criterion=CRITERION,
                max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=N_JOBS)
            model.fit(Xtr, ytr)
            acc = float(model.score(Xva, yva))
            mean_depth = float(np.mean([t.get_depth() for t in model.estimators_]))
            secs = time.time() - t0

            write_cell(ws, i, j, round(acc, 4))
            grid[i, j] = acc
            note = "  (depth saturated)" if mean_depth < max_depth - 0.5 else ""
            print(f"  n_est {n_est:>2}  max_depth {max_depth:>4} -> acc {acc * 100:6.2f}%   "
                  f"realised depth {mean_depth:5.1f}{note}   ({secs:.1f}s)", flush=True)

            results.append({"n_estimators": n_est, "max_depth": max_depth,
                            "accuracy": acc, "mean_tree_depth": round(mean_depth, 2),
                            "seconds": round(secs, 2)})

            if acc > best["acc"]:
                best = {"acc": acc, "n_estimators": n_est, "max_depth": max_depth}
                joblib.dump(model, OUTPUT_DIR / "rf_mfcc_best.joblib")
                np.save(OUTPUT_DIR / "rf_mfcc_best_importances.npy",
                        model.feature_importances_)
                print(f"    >> NEW BEST (n_est {n_est}, max_depth {max_depth})", flush=True)
        print("", flush=True)

    print("=" * 78, flush=True)
    print("validation accuracy grid (rows = n_estimators, columns = max_depth)", flush=True)
    print(f"  {'n_est':<7}" + "".join(f"{d:>7}" for d in SHEET_MAX_DEPTHS), flush=True)
    for i, n_est in enumerate(SHEET_ESTIMATORS):
        row = "".join(f"{grid[i, j] * 100:>7.2f}" for j in range(len(SHEET_MAX_DEPTHS)))
        flat = "  (flat)" if np.ptp(grid[i]) < 1e-12 else ""
        print(f"  {n_est:<7}{row}{flat}", flush=True)
    print("=" * 78, flush=True)

    col_var = [float(np.ptp(grid[:, j])) for j in range(len(SHEET_MAX_DEPTHS))]
    same_as_last = [j for j in range(1, len(SHEET_MAX_DEPTHS))
                    if np.allclose(grid[:, j], grid[:, j - 1])]
    if same_as_last:
        first = SHEET_MAX_DEPTHS[same_as_last[0]]
        print(f"\ncolumns from max_depth {first} onward duplicate the previous column -", flush=True)
        print(f"trees had already stopped growing, so the depth cap was not binding.", flush=True)

    print(f"\nbest accuracy per n_estimators:", flush=True)
    for n_est, a in zip(SHEET_ESTIMATORS, grid.max(axis=1)):
        print(f"  n_est {n_est:<3} {a * 100:6.2f}%", flush=True)

    print(f"\nBEST: n_estimators {best['n_estimators']}, max_depth {best['max_depth']}, "
          f"val accuracy {best['acc'] * 100:.2f}%", flush=True)
    print(f"  model -> {OUTPUT_DIR / 'rf_mfcc_best.joblib'}", flush=True)
    print(f"  grid completed in {(time.time() - grid_t0) / 60:.1f} min", flush=True)

    (OUTPUT_DIR / "grid_results.json").write_text(json.dumps({
        "model": "random_forest", "representation": "mfcc",
        "criterion": CRITERION, "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE, "classes": classes,
        "n_features": int(Xtr.shape[1]),
        "n_train": int(Xtr.shape[0]), "n_validation": int(Xva.shape[0]),
        "unconstrained_depth": {"min": int(min(depths)), "max": sat,
                                "mean": round(float(np.mean(depths)), 2)},
        "n_estimators_values": SHEET_ESTIMATORS,
        "max_depth_values": SHEET_MAX_DEPTHS,
        "best": {"n_estimators": best["n_estimators"], "max_depth": best["max_depth"],
                 "val_accuracy": best["acc"]},
        "results": results,
    }, indent=2))
    print(f"  grid log -> {OUTPUT_DIR / 'grid_results.json'}", flush=True)
    print("\nTHE TASK HAS BEEN COMPLETED.", flush=True)


if __name__ == "__main__":
    main()
