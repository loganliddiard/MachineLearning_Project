import os
from PIL import Image

# Root dataset directory
data_dir = "UDOT WINTER ROAD CONDITIONS.v1i.folder"

# Splits
dirs = [
    os.path.join(data_dir, "train"),
    os.path.join(data_dir, "valid"),
    os.path.join(data_dir, "test"),
]

# Target size
TARGET_SIZE = (224, 224)

# Supported image extensions
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resize_image(img_path, overwrite=True):
    try:
        with Image.open(img_path) as img:

            # Convert to RGB to avoid grayscale / RGBA issues
            img = img.convert("RGB")

            # Resize with high-quality resampling
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            if overwrite:
                img.save(img_path)
            else:
                # Save to new file if desired
                base, ext = os.path.splitext(img_path)
                new_path = f"{base}_224{ext}"
                img.save(new_path)

            return True

    except Exception as e:
        print(f"Failed: {img_path} | {e}")
        return False


def process_split(split_dir):
    print(f"\nProcessing: {split_dir}")

    total = 0
    success = 0

    for class_name in os.listdir(split_dir):

        class_dir = os.path.join(split_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):

            if filename.lower().endswith(IMG_EXTENSIONS):

                img_path = os.path.join(class_dir, filename)

                total += 1

                if resize_image(img_path):
                    success += 1

    print(f"Resized {success}/{total} images")


# Run processing
for split_dir in dirs:
    process_split(split_dir)

print("\nDone.")
