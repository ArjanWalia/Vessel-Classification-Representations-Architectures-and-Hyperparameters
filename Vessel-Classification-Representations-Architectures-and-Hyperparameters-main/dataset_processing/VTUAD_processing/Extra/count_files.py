from pathlib import Path

ROOT = ""
split = ROOT + "/" + "test"
classes = ["background", "cargo", "passengership", "tanker", "tug"]

count = 0
for cls in classes:
    cls_path = Path(split + "/" + cls)
    for wavs in cls_path.iterdir():
        count += 1
    print(f"{cls} : {count}")

    count = 0

print("\nDone")


