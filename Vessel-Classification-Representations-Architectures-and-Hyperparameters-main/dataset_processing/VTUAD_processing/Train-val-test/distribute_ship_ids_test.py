#This script partitions the remaining ship ids within each class into a test dataset. It draws
#only from ships/files not already consumed by the train and validation builds,
#completing a leakage-free three-way split.

import random
import shutil
from pathlib import Path


OUT = Path("")
TEST_DIR = OUT / "test"
SEED = 42
DIVISOR = 1                        # final split

TARGETS = {                        # required clip count per class in the test set
    "background": 40,
    "cargo": 40,
    "passengership": 40,
    "tanker": 40,
    "tug": 40,
}
BACKGROUND_CLASS = "background"    # one ship id


def list_ship_dirs(class_dir):
    """Unused ship_id_* folders in a class folder (skips ones already marked _used)."""
    if not class_dir.is_dir():
        return []
    ships = [d for d in sorted(class_dir.iterdir())
             if d.is_dir() and d.name.startswith("ship_id_") and not d.name.endswith("_used")]
    return ships


def wavs_in(ship_dir):
    """Unused wavs in a ship folder (skips files already marked _used)."""
    return sorted(p for p in ship_dir.iterdir()
                  if p.suffix.lower() == ".wav" and not p.stem.endswith("_used"))


def mark_file_used(p):
    """Rename a wav to <stem>_used.wav (records what this split consumed)."""
    target = p.with_name(p.stem + "_used.wav")
    if not target.exists():
        p.rename(target)


def build_background(class_dir, dest_dir, target, rng):
    """Background has a single ship id, so its FILES are partitioned instead of its ships.
    Train and validation already marked their thirds/halves _used; test uses ALL remaining
    files, draws the target from them, and marks them _used."""
    ships = list_ship_dirs(class_dir)
    if not ships:
        print(f"[background] no available ship folders in {class_dir} -> skip", flush=True)
        return 0

    all_wavs = []
    for s in ships:
        all_wavs.extend(wavs_in(s))          
    if not all_wavs:
        print(f"[background] no unused wavs left in {class_dir} -> skip", flush=True)
        return 0

    pool_size = len(all_wavs) // DIVISOR      
    if pool_size < 1:
        pool_size = 1
    pool = rng.sample(all_wavs, pool_size)
    rng.shuffle(pool)

    print(f"[background] {len(ships)} ship id(s), {len(all_wavs)} unused clips -> "
          f"file pool (all) {pool_size}, target {target}", flush=True)

    take = min(target, len(pool))
    for p in pool[:take]:
        shutil.copy2(str(p), str(dest_dir / p.name))

    for p in pool:
        mark_file_used(p)

    status = "OK" if take == target else f"SHORT by {target - take}"
    print(f"    copied {take}/{target} clips from the file pool "
          f"({pool_size} files marked _used) [{status}]", flush=True)
    if take < target:
        print(f"    !! only {pool_size} clips remained (< {target}); this was the last split",
              flush=True)
    return take


def main():
    rng = random.Random(SEED)
    if not OUT.is_dir():
        raise SystemExit(f"source outputs dir not found: {OUT}")
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SEED = {SEED}\n", flush=True)

    grand_total = 0
    for cls, target in TARGETS.items():
        class_dir = OUT / f"{cls}_ship_ids"
        dest_dir = TEST_DIR / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        # background: single ship id
        if cls == BACKGROUND_CLASS:
            grand_total += build_background(class_dir, dest_dir, target, rng)
            continue

        ships = list_ship_dirs(class_dir)     
        if not ships:
            print(f"[{cls}] no available ship folders in {class_dir} -> skip", flush=True)
            continue

        # how many ships to use for test
        n_ships = len(ships) // DIVISOR 
        if n_ships < 1:
            n_ships = 1
        n_ships = min(n_ships, len(ships))

        chosen = rng.sample(ships, n_ships)            # random ship selection using random seed
        quota = target // n_ships

        # pre-load and shuffle each chosen ship's clips
        pool = {}
        for s in chosen:
            w = wavs_in(s)
            rng.shuffle(w)
            pool[s] = w

        print(f"[{cls}] {len(ships)} unused ships -> using {n_ships} "
              f"(floor quota {quota}/ship), target {target}", flush=True)

        copied = 0
        taken = {s: 0 for s in chosen}

        # pass 1: take the floor quota from each chosen ship
        for s in chosen:
            avail = pool[s]
            take = min(quota, len(avail), target - copied)
            for p in avail[taken[s]: taken[s] + take]:
                shutil.copy2(str(p), str(dest_dir / p.name))
            taken[s] += take
            copied += take
            if copied >= target:
                break

        # pass 2: fill the remaining shortfall from the other chosen ships,
        #We take whatever is left in other ship ids until target is met
        if copied < target:
            for s in chosen:
                if copied >= target:
                    break
                avail = pool[s]
                remaining_in_ship = avail[taken[s]:]
                need = target - copied
                take = min(need, len(remaining_in_ship))
                for p in remaining_in_ship[:take]:
                    shutil.copy2(str(p), str(dest_dir / p.name))
                taken[s] += take
                copied += take

        # mark each consumed ship folder as used
        for s in chosen:
            used_name = s.parent / f"{s.name}_used"
            if not used_name.exists():
                s.rename(used_name)

        status = "OK" if copied == target else f"SHORT by {target - copied}"
        print(f"    copied {copied}/{target} clips from {n_ships} ships "
              f"[{status}]", flush=True)
        if copied < target:
            print(f"    !! not enough clips across the {n_ships} remaining ships to "
                  f"reach {target}; this was the last split", flush=True)
        grand_total += copied

    print(f"\nTEST dataset written to {TEST_DIR}")
    print(f"total clips copied: {grand_total}")


if __name__ == "__main__":
    main()