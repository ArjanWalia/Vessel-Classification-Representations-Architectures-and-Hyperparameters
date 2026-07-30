# This script constructs a split within each class of each ship ID to prevent leakage


import csv
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path("")
OUT = ROOT / "outputs_2"
SPLITS = ["train", "validation", "test"]
MOVE = False                      
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def mmsi_to_folder(mmsi_raw): #Normalize MMSI to integer for clean folder name
    s = str(mmsi_raw).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s.replace("/", "_")


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"root not found: {ROOT}")

    # Key: Obtain (MMSI, Class)
    # Value: list of tuples: (split order, file index, and source_wav_path)
    groups = defaultdict(list)
    total_rows = 0
    missing = []

    for split in SPLITS:
        csv_path = ROOT / split / f"metadata_{split}.csv"
        audio_dir = ROOT / split / "audio"
        if not csv_path.is_file():
            raise SystemExit(f"metadata not found: {csv_path}")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                cls = row["label"].strip()
                mmsi = mmsi_to_folder(row["MMSI"])
                fidx = row["file_index"].strip()
                src = audio_dir / cls / f"{fidx}.wav"
                if not src.is_file():
                    missing.append(str(src))
                    continue
                try:
                    order_key = (SPLIT_ORDER[split], int(fidx))
                except ValueError:
                    order_key = (SPLIT_ORDER[split], fidx)
                groups[(cls, mmsi)].append((order_key, src))

    print(f"read {total_rows} rows across {len(SPLITS)} splits")
    print(f"found {len(groups)} (class, ship) groups")
    if missing:
        print(f"WARNING: {len(missing)} rows had no matching .wav on disk "
              f"(first few): {missing[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)

    
    per_class_ships = defaultdict(int)
    per_class_clips = defaultdict(int)
    grand_copied = 0

    for (cls, mmsi), items in sorted(groups.items()):
        items.sort(key=lambda x: x[0])                 
        dest_dir = OUT / f"{cls}_ship_ids" / f"ship_id_{mmsi}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for new_idx, (_, src) in enumerate(items):
            dest = dest_dir / f"{mmsi}_{new_idx}.wav"
            if MOVE:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))
            grand_copied += 1
        per_class_ships[cls] += 1
        per_class_clips[cls] += len(items)

    print("\n=== summary (per class) ===")
    print(f"{'class':16}{'ships':>7}{'clips':>9}")
    for cls in sorted(per_class_clips):
        print(f"{cls:16}{per_class_ships[cls]:>7}{per_class_clips[cls]:>9}")
    print("-" * 32)
    print(f"{'TOTAL':16}{sum(per_class_ships.values()):>7}{grand_copied:>9}")
    print(f"\n{'moved' if MOVE else 'copied'} {grand_copied} wav files -> {OUT}")


if __name__ == "__main__":
    main()