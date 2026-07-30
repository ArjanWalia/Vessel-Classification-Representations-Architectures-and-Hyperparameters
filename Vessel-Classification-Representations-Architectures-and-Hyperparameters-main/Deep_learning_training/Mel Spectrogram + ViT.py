

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as VT
from transformers import ViTForImageClassification

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


BASE = Path("/content/drive/MyDrive/vessel_classification_final")
DATA_TAR = BASE / "Dataset" / "mel_dataset.tar"
GSHEET_PATH = BASE / "Documentation" / "Testing.gsheet"
OUTPUT_DIR = BASE / "Codes" / "Testing" / "Mel Spectrogram + ViT" / "Outputs"

SHEET_TAB = "Mel Spectrogram + ViT"
GSHEET_TITLE_FALLBACK = "Testing"
CREDENTIALS_JSON = "credentials.json"

LOCAL_DATA = Path("/content/mel_data")
MODEL_NAME = "google/vit-base-patch16-224"


OPTIMIZER = "adamw"
BATCH = 64
WEIGHT_DECAY = 1e-5
NO_AUGMENTATION = True

SHEET_LRS = [0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001, 0.000001]
SHEET_EPOCHS = [10, 20, 30, 40, 50]

START_LR = 0.05
START_EPOCHS = 10

PRIOR_RESULTS = {}

IMG_SIZE = 224
NUM_WORKERS = 2
VIT_MEAN, VIT_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def sh(cmd):
    if subprocess.call(cmd, shell=True) != 0:
        raise RuntimeError(f"command failed: {cmd}")


def lr_tag(lr):
    return f"{lr:.0e}"



def stage_dataset():
    def split_root(base):
        base = Path(base)
        if not base.exists():
            return None
        for d in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
            if (d / "train").is_dir() and (d / "validation").is_dir() \
                    and next((d / "validation").rglob("*.npy"), None) is not None:
                return d
        return None

    root = split_root(LOCAL_DATA)
    if root is not None:
        print(f"dataset already staged at {root}", flush=True)
        return root

    if not DATA_TAR.is_file():
        raise SystemExit(f"dataset tar not found: {DATA_TAR}")
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    print(f"copying + unpacking {DATA_TAR.name} to local disk...", flush=True)
    t0 = time.time()
    sh(f'cp "{DATA_TAR}" /content/_mel.tar')
    sh(f'tar -xf /content/_mel.tar -C "{LOCAL_DATA}"')
    sh("rm -f /content/_mel.tar")
    print(f"  unpacked in {time.time() - t0:.0f}s", flush=True)

    root = split_root(LOCAL_DATA)
    if root is None:
        raise SystemExit(f"could not find train/validation splits under {LOCAL_DATA}")
    return root


class MelDataset(Dataset):

    def __init__(self, split_dir, class_to_idx):
        self.samples = []
        for cls, idx in class_to_idx.items():
            cdir = split_dir / cls
            if not cdir.is_dir():
                print(f"  !! {split_dir.name}: no folder for class '{cls}'", flush=True)
                continue
            for npy in sorted(cdir.glob("*.npy")):
                self.samples.append((npy, idx))
        self.resize = VT.Resize((IMG_SIZE, IMG_SIZE), antialias=True)
        self.normalize = VT.Normalize(VIT_MEAN, VIT_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        arr = np.load(path).astype(np.float32)
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return self.normalize(self.resize(t)), label


def build_loaders():
    root = stage_dataset()
    classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_ds = MelDataset(root / "train", class_to_idx)
    val_ds = MelDataset(root / "validation", class_to_idx)
    print(f"classes: {classes}", flush=True)
    print(f"train clips {len(train_ds)} | validation clips {len(val_ds)}\n", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    return train_loader, val_loader, classes



def build_model(num_classes):
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True)
    return model.to(DEVICE)


def forward(model, x):
    return model(pixel_values=x).logits


@torch.no_grad()
def validate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = forward(model, x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_config(train_loader, val_loader, lr, epochs, num_classes):
    model = build_model(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()                       # no label smoothing
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    acc, state = 0.0, None
    try:
        for epoch in range(epochs):
            model.train()
            run_loss, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    loss = criterion(forward(model, x), y)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                run_loss += loss.item() * x.size(0)
                seen += x.size(0)

            acc = validate_accuracy(model, val_loader)
            print(f"      epoch {epoch + 1:>3}/{epochs}  "
                  f"train_loss {run_loss / max(seen, 1):.4f}  val_acc {acc * 100:.2f}%",
                  flush=True)

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    finally:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return acc, state



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
                                       rows=max(10, len(SHEET_LRS) + 2),
                                       cols=max(8, len(SHEET_EPOCHS) + 2))
        hdr_end = rowcol_to_a1(1, 1 + len(SHEET_EPOCHS))
        _update(ws, f"A1:{hdr_end}", [["Epochs / LR"] + SHEET_EPOCHS])
        lab_end = rowcol_to_a1(1 + len(SHEET_LRS), 1)
        _update(ws, f"A2:{lab_end}", [[lr] for lr in SHEET_LRS])
        return ws


def write_cell(ws, lr_index, epoch_index, value):
    from gspread.utils import rowcol_to_a1
    cell = rowcol_to_a1(lr_index + 2, epoch_index + 2)
    _update(ws, f"{cell}:{cell}", [[value]])


def save_best(state, classes, lr, epochs, acc):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "vit_mel_best.pt"
    torch.save({"model": "vit", "model_name": MODEL_NAME,
                "representation": "mel", "classes": classes,
                "normalization": {"mean": VIT_MEAN, "std": VIT_STD},
                "hyperparameters": {"lr": lr, "epochs": epochs, "batch": BATCH,
                                    "optimizer": OPTIMIZER, "weight_decay": WEIGHT_DECAY,
                                    "augmentation": not NO_AUGMENTATION},
                "val_accuracy": acc, "model_state": state}, path)
    return path



def main():
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"ViT ({MODEL_NAME}) + mel-spectrogram", flush=True)
    print(f"optimizer {OPTIMIZER} | batch {BATCH} | wd {WEIGHT_DECAY:.0e} | "
          f"no augmentation | no early stopping", flush=True)
    print(f"normalisation: mean {VIT_MEAN[0]} / std {VIT_STD[0]} (ViT, not ImageNet)",
          flush=True)
    total_epochs = len(SHEET_LRS) * sum(SHEET_EPOCHS)
    print(f"grid: {len(SHEET_LRS)} LRs x {SHEET_EPOCHS} epochs = "
          f"{len(SHEET_LRS) * len(SHEET_EPOCHS)} cells, {total_epochs} epochs total\n",
          flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spreadsheet = open_sheet()
    ws = get_grid_ws(spreadsheet)
    print(f"sheet '{spreadsheet.title}' tab '{SHEET_TAB}'\n", flush=True)

    train_loader, val_loader, classes = build_loaders()
    num_classes = len(classes)

    if PRIOR_RESULTS:
        pk = max(PRIOR_RESULTS, key=PRIOR_RESULTS.get)
        best = {"acc": PRIOR_RESULTS[pk], "lr": pk[0], "epochs": pk[1], "state": None}
        print(f"seeded best from PRIOR_RESULTS: lr {best['lr']:g}, "
              f"epochs {best['epochs']}, val_acc {best['acc']:.4f}\n", flush=True)
    else:
        best = {"acc": -1.0, "lr": None, "epochs": 0, "state": None}

    print(f"resuming at lr {START_LR:g}, epochs {START_EPOCHS}\n", flush=True)
    started = False
    grid_t0 = time.time()

    for li, lr in enumerate(SHEET_LRS):
        for ej, epochs in enumerate(SHEET_EPOCHS):
            if not started:
                if abs(lr - START_LR) <= 1e-12 and epochs == START_EPOCHS:
                    started = True
                else:
                    continue
            print("#" * 72, flush=True)
            print(f"lr {lr:g} | epochs {epochs}  (fresh model)", flush=True)
            print("#" * 72, flush=True)

            t0 = time.time()
            acc, state = train_one_config(train_loader, val_loader, lr, epochs, num_classes)
            mins = (time.time() - t0) / 60

            write_cell(ws, li, ej, round(acc, 4))
            print(f"  >> val_acc {acc * 100:.2f}%  ({mins:.1f} min)  "
                  f"-> sheet row {li + 2}, col {ej + 2}", flush=True)

            if state is not None and acc > best["acc"]:
                best = {"acc": acc, "lr": lr, "epochs": epochs, "state": state}
                path = save_best(state, classes, lr, epochs, acc)
                print(f"  >> NEW BEST -> {path}", flush=True)

            elapsed = (time.time() - grid_t0) / 3600
            print(f"  >> elapsed this session: {elapsed:.2f} h\n", flush=True)
            time.sleep(0.3)

    if not started:
        print(f"!! resume point (lr {START_LR:g}, epochs {START_EPOCHS}) is not in the "
              f"grid - nothing ran", flush=True)
        return

    print("=" * 60, flush=True)
    if best["lr"] is None:
        print("no successful cell; nothing saved", flush=True)
    elif best["state"] is not None:
        print(f"BEST: lr {best['lr']:g}, epochs {best['epochs']}, "
              f"val_acc {best['acc'] * 100:.2f}%", flush=True)
        print(f"  weights -> {OUTPUT_DIR / 'vit_mel_best.pt'}", flush=True)
    else:
        print(f"BEST is a cell from an earlier session: lr {best['lr']:g}, "
              f"epochs {best['epochs']} (val_acc {best['acc']:.4f}).", flush=True)
        print("retraining it once to produce its weights...", flush=True)
        acc, state = train_one_config(train_loader, val_loader,
                                      best["lr"], best["epochs"], num_classes)
        if state is not None:
            path = save_best(state, classes, best["lr"], best["epochs"], acc)
            print(f"  retrained val_acc {acc * 100:.2f}%  -> {path}", flush=True)
    print("=" * 60, flush=True)
    print("\nTHE TASK HAS BEEN COMPLETED.", flush=True)


if __name__ == "__main__":
    main()
