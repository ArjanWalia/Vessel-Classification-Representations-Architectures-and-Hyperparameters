

import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np


BASE = Path("/content/drive/MyDrive/vessel_classification_final")
DATA_TAR = BASE / "Dataset" / "mfcc_dataset.tar"
GSHEET_PATH = BASE / "Documentation" / "Testing.gsheet"
OUTPUT_DIR = BASE / "Codes" / "Testing" / "MFCC + Logistic Regression" / "Outputs"

SHEET_TAB = "MFCC + Logistic Regression"
GSHEET_TITLE_FALLBACK = "Testing"
CREDENTIALS_JSON = "credentials.json"

LOCAL_DATA = Path("/content/mfcc_data")


SHEET_CS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
SHEET_ITERS = [50, 100, 150, 200, 250, 300]

SOLVER = "lbfgs"
PENALTY = "l2"
RANDOM_STATE = 42
N_JOBS = -1



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")



def find_data_root(base):
    """Locate the folder holding X_train.npy / X_validation.npy."""
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
    X = np.load(root / f"X_{split}.npy").astype(np.float64)
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
                                       rows=max(10, len(SHEET_CS) + 2),
                                       cols=max(8, len(SHEET_ITERS) + 2))
        hdr_end = rowcol_to_a1(1, 1 + len(SHEET_ITERS))
        _update(ws, f"A1:{hdr_end}", [["Iterations/C"] + SHEET_ITERS])
        lab_end = rowcol_to_a1(1 + len(SHEET_CS), 1)
        _update(ws, f"A2:{lab_end}", [[c] for c in SHEET_CS])
        return ws


def write_cell(ws, c_index, iter_index, value):
    from gspread.utils import rowcol_to_a1
    cell = rowcol_to_a1(c_index + 2, iter_index + 2)
    _update(ws, f"{cell}:{cell}", [[value]])



def main():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        import joblib
    except ImportError:
        sys.exit("this script needs scikit-learn and joblib:  pip install scikit-learn joblib")

    print("Logistic Regression + MFCC (mean+std aggregated)", flush=True)
    print(f"solver {SOLVER} | penalty {PENALTY} | multinomial", flush=True)
    print(f"grid: C {SHEET_CS} x max_iter {SHEET_ITERS} = "
          f"{len(SHEET_CS) * len(SHEET_ITERS)} cells\n", flush=True)

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
    print(f"classes: {classes}", flush=True)
    print(f"feature check: train mean {Xtr.mean():+.4f}, std {Xtr.std():.4f} "
          f"(already standardized -> not re-scaled here)\n", flush=True)

    best = {"acc": -1.0, "C": None, "iters": None}
    results = []
    grid = np.full((len(SHEET_CS), len(SHEET_ITERS)), np.nan)
    grid_t0 = time.time()

    for i, C in enumerate(SHEET_CS):
        for j, max_iter in enumerate(SHEET_ITERS):
            t0 = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(
                    C=C, max_iter=max_iter, solver=SOLVER, penalty=PENALTY,
                    random_state=RANDOM_STATE, n_jobs=N_JOBS)
                model.fit(Xtr, ytr)
                hit_warning = any(issubclass(w.category, ConvergenceWarning)
                                  for w in caught)

            acc = float(model.score(Xva, yva))
            n_iter = int(np.max(model.n_iter_))
            converged = (n_iter < max_iter) and not hit_warning
            secs = time.time() - t0

            write_cell(ws, i, j, round(acc, 4))
            grid[i, j] = acc
            flag = "converged" if converged else "HIT CAP"
            print(f"  C {C:<8g} max_iter {max_iter:<4} -> acc {acc * 100:6.2f}%   "
                  f"n_iter {n_iter:>4}/{max_iter:<4} {flag:<10} ({secs:.1f}s)", flush=True)

            results.append({"C": C, "max_iter": max_iter, "accuracy": acc,
                            "n_iter": n_iter, "converged": converged,
                            "seconds": round(secs, 2)})

            if acc > best["acc"]:
                best = {"acc": acc, "C": C, "iters": max_iter}
                joblib.dump(model, OUTPUT_DIR / "logreg_mfcc_best.joblib")
                np.save(OUTPUT_DIR / "logreg_mfcc_best_coef.npy", model.coef_)
                np.save(OUTPUT_DIR / "logreg_mfcc_best_intercept.npy", model.intercept_)
                print(f"    >> NEW BEST (C {C:g}, max_iter {max_iter})", flush=True)
        print("", flush=True)

    print("=" * 68, flush=True)
    print("validation accuracy grid (rows = C, columns = max_iter)", flush=True)
    print(f"  {'C':<10}" + "".join(f"{it:>9}" for it in SHEET_ITERS), flush=True)
    for i, C in enumerate(SHEET_CS):
        row = "".join(f"{grid[i, j] * 100:>9.2f}" for j in range(len(SHEET_ITERS)))
        flat = "  (flat)" if np.ptp(grid[i]) < 1e-12 else ""
        print(f"  {C:<10g}{row}{flat}", flush=True)
    print("=" * 68, flush=True)

    n_flat = sum(1 for i in range(len(SHEET_CS)) if np.ptp(grid[i]) < 1e-12)
    print(f"\n{n_flat}/{len(SHEET_CS)} C-rows are flat across max_iter - in those the "
          f"solver converged\nbefore the smallest cap, so only C affected the result.",
          flush=True)
    best_c_curve = grid.max(axis=1)
    print(f"\nbest accuracy per C:", flush=True)
    for C, a in zip(SHEET_CS, best_c_curve):
        print(f"  C {C:<10g} {a * 100:6.2f}%", flush=True)

    print(f"\nBEST: C {best['C']:g}, max_iter {best['iters']}, "
          f"val accuracy {best['acc'] * 100:.2f}%", flush=True)
    print(f"  model -> {OUTPUT_DIR / 'logreg_mfcc_best.joblib'}", flush=True)
    print(f"  grid completed in {(time.time() - grid_t0) / 60:.1f} min", flush=True)

    (OUTPUT_DIR / "grid_results.json").write_text(json.dumps({
        "model": "logistic_regression", "representation": "mfcc",
        "solver": SOLVER, "penalty": PENALTY, "random_state": RANDOM_STATE,
        "classes": classes, "n_features": int(Xtr.shape[1]),
        "n_train": int(Xtr.shape[0]), "n_validation": int(Xva.shape[0]),
        "C_values": SHEET_CS, "max_iter_values": SHEET_ITERS,
        "best": {"C": best["C"], "max_iter": best["iters"], "val_accuracy": best["acc"]},
        "results": results,
    }, indent=2))
    print(f"  grid log -> {OUTPUT_DIR / 'grid_results.json'}", flush=True)
    print("\nTHE TASK HAS BEEN COMPLETED.", flush=True)


if __name__ == "__main__":
    main()
