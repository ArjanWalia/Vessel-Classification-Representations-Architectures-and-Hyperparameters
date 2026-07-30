import argparse
import os
import wave

import pandas as pd

from tqdm import tqdm

from config import WORK_DIR, INCLUSION_RADIUS, METADATA_SECONDS
from utils import bcolors, create_dir, get_exclusion_radius
from group_by_mmsi import BACKGROUND_LABEL, GROUPED_DIR


SPLITS_DIR = "11_dataset_splits"
USED_MARKER = "_used"

# The number of segments that each class contributes to every split.
SPLIT_QUOTAS = {
    "train": {
        "background": 1376,
        "cargo": 688,
        "passengership": 688,
        "tanker": 688,
        "tug": 688,
    },
    "validation": {
        "background": 144,
        "cargo": 72,
        "passengership": 72,
        "tanker": 72,
        "tug": 72,
    },
    "test": {
        "background": 80,
        "cargo": 40,
        "passengership": 40,
        "tanker": 40,
        "tug": 40,
    },
}

# How many partitions the available units are divided into for every split. The
# split is drawn from the first partition and the rest is kept for the splits
# that are built afterwards.
SPLIT_PARTITIONS = {
    "train": 3,
    "validation": 2,
    # The test split is the last one, so there is nothing left to reserve.
    "test": 1,
}


def is_used(name):
    # The marker sits on the folder name for vessels and before the extension
    # for the background clips.
    return os.path.splitext(name)[0].endswith(USED_MARKER)


def get_sort_key(unit):
    """Sort the units numerically so the partitions are reproducible."""
    stem = os.path.splitext(os.path.basename(unit))[0]
    return (0, int(stem)) if stem.isdigit() else (1, unit)


def get_units_from_manifest(manifest, metadata):
    """Build the pool of drawable units for every class.

    The vessel classes are drawn from their MMSI folders. The background class
    has a single MMSI folder, so it is partitioned by clip instead.
    """
    segments = (
        metadata.sort_values(["path", "sub_init"])
        .groupby("path")
        .sub_init.apply(list)
        .to_dict()
    )

    units = {}
    for label, rows in manifest.groupby("label"):
        pool = {}
        for row in rows.itertuples(index=False):
            if any(is_used(part) for part in row.path.split(os.sep)):
                continue

            unit = os.path.basename(row.path) if label == BACKGROUND_LABEL else row.mmsi
            for sub_init in segments.get(row.source_path, []):
                pool.setdefault(str(unit), []).append((row.path, sub_init))

        units[label] = dict(sorted(pool.items(), key=lambda item: get_sort_key(item[0])))

    return units


def draw_evenly(pool, ordered_units, quota, cursors, drawn):
    """Take one segment at a time from each unit in turn until the quota is met."""
    while len(drawn) < quota:
        progressed = False
        for unit in ordered_units:
            if len(drawn) >= quota:
                break
            cursor = cursors[unit]
            if cursor < len(pool[unit]):
                path, sub_init = pool[unit][cursor]
                drawn.append((unit, path, sub_init))
                cursors[unit] = cursor + 1
                progressed = True
        if not progressed:
            break

    return drawn


def draw_from_class(pool, quota, partition_count):
    """Draw the quota from the first partition of units, borrowing if needed."""
    names = list(pool.keys())
    size = len(names) // partition_count
    partitions = [names[index * size : (index + 1) * size] for index in range(partition_count - 1)]
    partitions.append(names[(partition_count - 1) * size :])

    cursors = {unit: 0 for unit in names}
    drawn = []
    ordered = []
    for index, partition in enumerate(partitions):
        if not partition:
            continue
        ordered = ordered + partition
        draw_evenly(pool, ordered, quota, cursors, drawn)
        if len(drawn) >= quota:
            break
        if index < len(partitions) - 1:
            print(
                f"{bcolors.WARNING}  quota not met from partition {index + 1}, "
                f"borrowing from partition {index + 2}{bcolors.ENDC}"
            )

    return drawn, size


def save_segments(grouped_directory, output_directory, label, drawn, seconds):
    """Cut and save every drawn segment, opening each source clip only once."""
    by_clip = {}
    for unit, path, sub_init in drawn:
        by_clip.setdefault((unit, path), []).append(sub_init)

    saved = []
    for (unit, path), sub_inits in by_clip.items():
        source_file = os.path.join(grouped_directory, path)
        destination_directory = create_dir(
            create_dir(output_directory, label), unit.replace(".wav", "")
        )
        file_name = os.path.splitext(os.path.basename(path))[0]

        with wave.open(source_file, "rb") as source_wav:
            frame_rate = source_wav.getframerate()
            segment_frames = frame_rate * seconds

            for sub_init in sorted(sub_inits):
                source_wav.setpos(sub_init * frame_rate)
                frames = source_wav.readframes(segment_frames)

                destination_file = os.path.join(
                    destination_directory, f"{file_name}_{sub_init:05d}.wav"
                )
                with wave.open(destination_file, "wb") as destination_wav:
                    destination_wav.setnchannels(source_wav.getnchannels())
                    destination_wav.setsampwidth(source_wav.getsampwidth())
                    destination_wav.setframerate(frame_rate)
                    destination_wav.writeframes(frames)

                saved.append(
                    {
                        "label": label,
                        "unit": unit,
                        "source_path": path,
                        "sub_init": sub_init,
                        "path": os.path.relpath(destination_file, output_directory),
                    }
                )

    return saved


def mark_used(grouped_directory, label, used_units):
    """Rename every unit that was drawn from so it is never reused."""
    renamed = {}
    for unit in used_units:
        if label == BACKGROUND_LABEL:
            matches = [
                os.path.join(root, unit)
                for root, _, files in os.walk(os.path.join(grouped_directory, label))
                if unit in files
            ]
            if not matches:
                continue
            current = matches[0]
            stem, extension = os.path.splitext(current)
            target = f"{stem}{USED_MARKER}{extension}"
        else:
            current = os.path.join(grouped_directory, label, unit)
            target = f"{current}{USED_MARKER}"

        if os.path.exists(current) and not os.path.exists(target):
            os.rename(current, target)
        renamed[os.path.relpath(current, grouped_directory)] = os.path.relpath(
            target, grouped_directory
        )

    return renamed


def build_split(
    grouped_directory, metadata_file, output_directory, quotas, partition_count, seconds
):
    manifest = pd.read_csv(os.path.join(grouped_directory, "grouped_manifest.csv"))
    manifest["mmsi"] = manifest.mmsi.astype(str)
    metadata = pd.read_csv(metadata_file)

    units = get_units_from_manifest(manifest, metadata)

    split = []
    renamed = {}
    report = []
    for label, quota in quotas.items():
        pool = units.get(label, {})
        if not pool:
            print(f"{bcolors.FAIL}No units available for {label}{bcolors.ENDC}")
            continue

        print(f"\n{bcolors.OKBLUE}{label}{bcolors.ENDC}: {len(pool)} units, quota {quota}")
        drawn, partition_size = draw_from_class(pool, quota, partition_count)
        if len(drawn) < quota:
            print(
                f"{bcolors.FAIL}  only {len(drawn)} of {quota} segments available"
                f"{bcolors.ENDC}"
            )

        used_units = sorted({unit for unit, _, _ in drawn}, key=get_sort_key)
        saved = save_segments(grouped_directory, output_directory, label, drawn, seconds)
        split.extend(saved)
        renamed.update(mark_used(grouped_directory, label, used_units))

        borrowed = len(used_units) - min(len(used_units), partition_size)
        print(
            f"  drew {len(drawn)} segments from {len(used_units)} units "
            f"(first partition = {partition_size}, borrowed = {borrowed})"
        )
        report.append(
            {
                "label": label,
                "quota": quota,
                "segments": len(drawn),
                "units_total": len(pool),
                "units_in_first_partition": partition_size,
                "units_used": len(used_units),
                "units_borrowed": borrowed,
                "segments_per_unit_min": min(
                    (sum(1 for u, _, _ in drawn if u == unit) for unit in used_units),
                    default=0,
                ),
                "segments_per_unit_max": max(
                    (sum(1 for u, _, _ in drawn if u == unit) for unit in used_units),
                    default=0,
                ),
            }
        )

    # Keep the grouped manifest pointing at the renamed folders.
    if renamed:
        manifest["path"] = [
            next((v for k, v in renamed.items() if p == k or p.startswith(k + os.sep)), p)
            for p in manifest.path
        ]
        manifest.to_csv(os.path.join(grouped_directory, "grouped_manifest.csv"), index=False)

    return pd.DataFrame(split), pd.DataFrame(report)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Build a dataset split from the MMSI grouped clips"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=list(SPLIT_QUOTAS.keys()),
        default="train",
        help="The split to build. The units it draws from are marked as used so "
        "that the splits built afterwards never reuse the same ship.",
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=None,
        help="How many partitions the available units are divided into. "
        "Defaults to the value configured for the chosen split.",
    )

    parser.add_argument("--work_dir", "-w", type=str, default=WORK_DIR)
    parser.add_argument("--inclusion_radius", type=int, default=INCLUSION_RADIUS)
    parser.add_argument(
        "--metadata_file", type=str, default=f"metadata_{METADATA_SECONDS}s.csv"
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=METADATA_SECONDS,
        help="The duration of each segment.",
    )

    return parser


def main():
    args = create_parser().parse_args()

    exclusion_radius = get_exclusion_radius(args.inclusion_radius)
    range_name = f"inclusion_{args.inclusion_radius}_exclusion_{exclusion_radius}"

    grouped_directory = os.path.join(args.work_dir, GROUPED_DIR, range_name)
    metadata_file = os.path.join(
        args.work_dir, "07b_classified_wav_files", range_name, args.metadata_file
    )
    splits_directory = create_dir(create_dir(args.work_dir, SPLITS_DIR), range_name)
    output_directory = create_dir(splits_directory, args.split)

    partition_count = args.partitions or SPLIT_PARTITIONS[args.split]

    print(f"\n{bcolors.HEADER}Building the {args.split} split{bcolors.ENDC}")
    print(f"Reading {grouped_directory}")
    print(f"Writing to {output_directory}")
    print(f"Dividing the available units into {partition_count} partitions")

    split, report = build_split(
        grouped_directory,
        metadata_file,
        output_directory,
        SPLIT_QUOTAS[args.split],
        partition_count,
        args.seconds,
    )

    manifest_file = os.path.join(splits_directory, f"{args.split}_manifest.csv")
    split.to_csv(manifest_file, index=False)
    report.to_csv(os.path.join(splits_directory, f"{args.split}_report.csv"), index=False)

    print(f"\n{report.to_string(index=False)}")
    print(f"\nTotal {args.split} segments: {len(split)}")
    print(f"Manifest saved to {manifest_file}")


if __name__ == "__main__":
    main()
