import argparse
import os
import shutil
import wave

import pandas as pd

from tqdm import tqdm

from config import WORK_DIR, INCLUSION_RADIUS, METADATA_SECONDS
from utils import bcolors, create_dir, get_exclusion_radius


BACKGROUND_LABEL = "background"
# The background class has no vessel identity, so every background clip is
# stored under a single MMSI folder.
BACKGROUND_MMSI = 0

GROUPED_DIR = "10_mmsi_grouped_clips"


def get_mmsi_folder_name(mmsi):
    return str(int(mmsi))


def get_clips_from_metadata(metadata_file):
    """Categorise every clip in the metadata by its class and MMSI.

    Returns one row per WAV file with the label, the MMSI folder name it
    belongs to and the 1 second offsets that the metadata lists for it.
    """
    metadata = pd.read_csv(metadata_file)

    clips = []
    for path, rows in metadata.groupby("path"):
        labels = rows.label.unique()
        mmsis = rows.MMSI.unique()

        if len(labels) > 1:
            print(f"{bcolors.WARNING}Skipping {path}: more than one label {labels}{bcolors.ENDC}")
            continue
        if len(mmsis) > 1:
            print(f"{bcolors.WARNING}Skipping {path}: more than one MMSI {mmsis}{bcolors.ENDC}")
            continue

        label = labels[0]
        mmsi = BACKGROUND_MMSI if label == BACKGROUND_LABEL else mmsis[0]

        clips.append(
            {
                "label": label,
                "mmsi": get_mmsi_folder_name(mmsi),
                "path": path,
                "sub_inits": sorted(rows.sub_init.tolist()),
            }
        )

    return clips


def place_wav_file(source_file, destination_file, mode):
    if os.path.exists(destination_file):
        os.remove(destination_file)

    if mode == "hardlink":
        os.link(source_file, destination_file)
    elif mode == "symlink":
        os.symlink(os.path.abspath(source_file), destination_file)
    else:
        shutil.copyfile(source_file, destination_file)


def save_wav_segments(source_file, destination_directory, sub_inits, seconds):
    """Cut the clip into the fixed length segments listed in the metadata."""
    saved = []
    file_name = os.path.splitext(os.path.basename(source_file))[0]

    with wave.open(source_file, "rb") as source_wav:
        frame_rate = source_wav.getframerate()
        segment_frames = frame_rate * seconds
        total_frames = source_wav.getnframes()

        for sub_init in sub_inits:
            start_frame = sub_init * frame_rate
            if start_frame + segment_frames > total_frames:
                print(
                    f"{bcolors.WARNING}Skipping {source_file} at {sub_init}s: "
                    f"beyond the end of the file{bcolors.ENDC}"
                )
                continue

            source_wav.setpos(start_frame)
            frames = source_wav.readframes(segment_frames)

            destination_file = os.path.join(
                destination_directory, f"{file_name}_{sub_init:05d}.wav"
            )
            with wave.open(destination_file, "wb") as destination_wav:
                destination_wav.setnchannels(source_wav.getnchannels())
                destination_wav.setsampwidth(source_wav.getsampwidth())
                destination_wav.setframerate(frame_rate)
                destination_wav.writeframes(frames)

            saved.append((destination_file, sub_init))

    return saved


def group_clips_by_mmsi(metadata_root, metadata_file, output_directory, unit, mode, seconds):
    clips = get_clips_from_metadata(os.path.join(metadata_root, metadata_file))
    print(f"Categorised {len(clips)} clips by class and MMSI")

    manifest = []
    for clip in tqdm(clips, total=len(clips)):
        # The metadata stores the clip paths relative to the metadata folder.
        source_file = os.path.join(metadata_root, clip["path"])
        if not os.path.exists(source_file):
            print(f"{bcolors.WARNING}Skipping {clip['path']}: file not found{bcolors.ENDC}")
            continue

        # Every MMSI folder lives inside the folder of its own class.
        label_directory = create_dir(output_directory, clip["label"])
        mmsi_directory = create_dir(label_directory, clip["mmsi"])

        if unit == "segment":
            saved = save_wav_segments(source_file, mmsi_directory, clip["sub_inits"], seconds)
        else:
            destination_file = os.path.join(mmsi_directory, os.path.basename(source_file))
            place_wav_file(source_file, destination_file, mode)
            saved = [(destination_file, None)]

        for destination_file, sub_init in saved:
            manifest.append(
                {
                    "label": clip["label"],
                    "mmsi": clip["mmsi"],
                    "group_id": f"{clip['label']}/{clip['mmsi']}",
                    "source_path": clip["path"],
                    "sub_init": sub_init,
                    "path": os.path.relpath(destination_file, output_directory),
                }
            )

    return pd.DataFrame(manifest)


def save_manifest(manifest, output_directory):
    manifest_file = os.path.join(output_directory, "grouped_manifest.csv")
    manifest.to_csv(manifest_file, index=False)

    summary = (
        manifest.groupby("label")
        .agg(mmsi_folders=("mmsi", "nunique"), clips=("path", "count"))
        .sort_values("clips", ascending=False)
    )
    summary_file = os.path.join(output_directory, "grouped_summary.csv")
    summary.to_csv(summary_file)

    print(f"\n{summary.to_string()}")
    print(f"\nTotal MMSI folders: {manifest.group_id.nunique()}")
    print(f"Manifest saved to {manifest_file}")
    print(f"Summary saved to {summary_file}")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Group the classified clips into class/MMSI folders"
    )

    parser.add_argument(
        "--work_dir",
        "-w",
        type=str,
        default=WORK_DIR,
        help="The path to the workspace directory.",
    )

    parser.add_argument(
        "--inclusion_radius",
        type=int,
        default=INCLUSION_RADIUS,
        help="The inclusion radius of the dataset that will be grouped.",
    )

    parser.add_argument(
        "--metadata_file",
        type=str,
        default=f"metadata_{METADATA_SECONDS}s.csv",
        help="The metadata file used to categorise the clips.",
    )

    parser.add_argument(
        "--unit",
        type=str,
        choices=["file", "segment"],
        default="file",
        help="Place the whole WAV files in the MMSI folders, or cut them into "
        "the fixed length segments listed in the metadata.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How the whole WAV files are placed. Only used when --unit is file.",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=METADATA_SECONDS,
        help="The duration of each segment. Only used when --unit is segment.",
    )

    return parser


def main():
    args = create_parser().parse_args()

    exclusion_radius = get_exclusion_radius(args.inclusion_radius)
    range_name = f"inclusion_{args.inclusion_radius}_exclusion_{exclusion_radius}"

    metadata_root = os.path.join(args.work_dir, "07b_classified_wav_files", range_name)
    grouped_directory = create_dir(args.work_dir, GROUPED_DIR)
    output_directory = create_dir(grouped_directory, range_name)

    print(f"\n{bcolors.HEADER}Grouping clips by class and MMSI{bcolors.ENDC}")
    print(f"Reading {os.path.join(metadata_root, args.metadata_file)}")
    print(f"Writing to {output_directory}")

    manifest = group_clips_by_mmsi(
        metadata_root,
        args.metadata_file,
        output_directory,
        args.unit,
        args.mode,
        args.seconds,
    )
    save_manifest(manifest, output_directory)


if __name__ == "__main__":
    main()
