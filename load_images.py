import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import os


# =========================
# SETTINGS
# =========================
IMAGE_SIZE = (50, 50)

TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.10
TEST_SPLIT = 0.10


# =========================
# LOAD IMAGES + LABELS
# =========================
def load_images_labels():

    images_labels = []

    image_paths = glob(
        "gestures/*/*.jpg"
    )

    image_paths.sort()

    if len(image_paths) == 0:

        raise Exception(
            "No gesture images found."
        )

    print(
        f"\nTotal images found: "
        f"{len(image_paths)}"
    )

    for image_path in image_paths:

        try:

            # LABEL
            label = os.path.basename(
                os.path.dirname(image_path)
            )

            # READ IMAGE
            img = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            # SKIP INVALID IMAGE
            if img is None:
                print(
                    f"Skipping invalid image: "
                    f"{image_path}"
                )
                continue

            # RESIZE IMAGE
            img = cv2.resize(
                img,
                IMAGE_SIZE
            )

            # NORMALIZE IMAGE
            img = img.astype(
                np.uint8
            )

            images_labels.append(
                (img, int(label))
            )

        except Exception as e:

            print(
                f"Error loading "
                f"{image_path}: {e}"
            )

    return images_labels


# =========================
# SAVE PICKLE FILE
# =========================
def save_pickle(filename, data):

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(
        f"{filename} saved successfully."
    )


# =========================
# MAIN
# =========================
def main():

    # LOAD DATA
    images_labels = load_images_labels()

    # SHUFFLE DATA
    images_labels = shuffle(
        images_labels,
        random_state=42
    )

    # SPLIT IMAGES + LABELS
    images, labels = zip(*images_labels)

    images = np.array(images)
    labels = np.array(labels)

    print(
        f"\nTotal dataset size: "
        f"{len(images)}"
    )

    # =========================
    # TRAIN + TEMP SPLIT
    # =========================
    train_images, temp_images, \
    train_labels, temp_labels = \
        train_test_split(
            images,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels
        )

    # =========================
    # VALIDATION + TEST SPLIT
    # =========================
    val_images, test_images, \
    val_labels, test_labels = \
        train_test_split(
            temp_images,
            temp_labels,
            test_size=0.50,
            random_state=42,
            stratify=temp_labels
        )

    # =========================
    # PRINT DATASET INFO
    # =========================
    print("\nDataset Split:")
    print(
        f"Train Images: {len(train_images)}"
    )

    print(
        f"Validation Images: "
        f"{len(val_images)}"
    )

    print(
        f"Test Images: "
        f"{len(test_images)}"
    )

    # =========================
    # SAVE FILES
    # =========================
    save_pickle(
        "train_images",
        train_images
    )

    save_pickle(
        "train_labels",
        train_labels
    )

    save_pickle(
        "val_images",
        val_images
    )

    save_pickle(
        "val_labels",
        val_labels
    )

    save_pickle(
        "test_images",
        test_images
    )

    save_pickle(
        "test_labels",
        test_labels
    )

    print(
        "\nAll dataset files created "
        "successfully."
    )


# =========================
# START
# =========================
if __name__ == "__main__":

    main()