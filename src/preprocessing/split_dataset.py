from pathlib import Path
import shutil
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "balanced_dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "split_dataset"

CLASSES = ["correct", "error"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def split_class(class_name):

    source = SOURCE_DIR / class_name

    images = list(source.glob("*.jpg"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\nClass: {class_name}")
    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:

        destination = OUTPUT_DIR / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(
                image,
                destination / image.name
            )


def main():

    print("\n==========================================")
    print("   PROCEVISION AI - DATASET SPLITTER")
    print("==========================================")

    for class_name in CLASSES:
        split_class(class_name)

    print("\n==========================================")
    print("DATASET SPLIT COMPLETE")
    print("==========================================")
    print(f"Output: {OUTPUT_DIR}")
    print("==========================================\n")


if __name__ == "__main__":
    main()