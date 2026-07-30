import os
import shutil

SOURCE_DATASETS = ['2000_4000', '3000_5000', '4000_6000', 'ONC']


def create_cumulative_dataset(source_root, output_root="cumulative_dataset"):
    splits = ['train', 'test', 'validation']

    for split in splits:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)

    dataset_folders = [d for d in SOURCE_DATASETS
                       if os.path.isdir(os.path.join(source_root, d))]
    missing = [d for d in SOURCE_DATASETS if d not in dataset_folders]
    if missing:
        print(f"WARNING: source folders not found: {missing}")
    if not dataset_folders:
        raise SystemExit(f"none of {SOURCE_DATASETS} found under {source_root}")

    print(f"Processing datasets: {dataset_folders}")

    total_copied = 0
    per_dataset = {}
    per_class = {}

    for dataset_idx, dataset_name in enumerate(dataset_folders):
        print(f"  Processing dataset {dataset_idx + 1}/{len(dataset_folders)}: "
              f"{dataset_name}")
        dataset_path = os.path.join(source_root, dataset_name)
        dataset_total = 0

        for split in splits:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                print(f"    Skipping split '{split}': path does not exist.")
                continue
            print(f"    Processing split: {split}")

            try:
                class_names = sorted(os.listdir(split_path))
            except Exception as e:
                print(f"      ERROR: could not list {split_path}: {e}")
                continue

            for class_name in class_names:
                class_dir = os.path.join(split_path, class_name)
                if not os.path.isdir(class_dir):
                    continue

                dest_class_path = os.path.join(output_root, split, class_name)
                os.makedirs(dest_class_path, exist_ok=True)

                try:
                    files_in_class = sorted(
                        f for f in os.listdir(class_dir)
                        if os.path.isfile(os.path.join(class_dir, f))
                        and os.path.splitext(f)[1].lower() == '.wav')
                except Exception as e:
                    print(f"        ERROR: could not list {class_dir}: {e}")
                    continue

                print(f"      Processing class: {class_name} "
                      f"({len(files_in_class)} files)")

                for file_index, filename in enumerate(files_in_class):
                    src_file = os.path.join(class_dir, filename)

                    if file_index % 500 == 0 and file_index > 0:
                        print(f"        {file_index}/{len(files_in_class)}")

                    clip_index = 0
                    extension = os.path.splitext(filename)[1]
                    new_name = (f"{dataset_name}_{split}_{class_name}_"
                                f"{file_index}_{clip_index}{extension}")
                    dst_file = os.path.join(dest_class_path, new_name)

                    if os.path.exists(dst_file) and os.path.getsize(dst_file) > 0:
                        continue
                    shutil.copy2(src_file, dst_file)

                n = len(files_in_class)
                dataset_total += n
                total_copied += n
                per_class[(split, class_name)] = per_class.get(
                    (split, class_name), 0) + n

        per_dataset[dataset_name] = dataset_total

    print()
    print("Files per dataset:")
    for name in dataset_folders:
        print(f"  {name:<12} {per_dataset.get(name, 0):>7}")

    classes = sorted({c for _, c in per_class})
    print()
    print(f"{'split':<12}" + "".join(f"{c:>16}" for c in classes) + f"{'total':>9}")
    for split in splits:
        row = "".join(f"{per_class.get((split, c), 0):>16}" for c in classes)
        print(f"{split:<12}{row}"
              f"{sum(per_class.get((split, c), 0) for c in classes):>9}")
    print(f"{'TOTAL':<12}"
          + "".join(f"{sum(per_class.get((s, c), 0) for s in splits):>16}"
                    for c in classes)
          + f"{total_copied:>9}")

    print()
    print(f"Successfully created cumulative dataset at: "
          f"{os.path.abspath(output_root)}")


root_source = "/content/drive/MyDrive/vessel_classification_final/Dataset"
create_cumulative_dataset(root_source)
