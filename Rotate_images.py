import cv2
import os
import random


# =========================
# SETTINGS
# =========================
GESTURE_FOLDER = "gestures"

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)


# =========================
# FLIP IMAGES
# =========================
def flip_images():

    # CHECK GESTURE FOLDER
    if not os.path.exists(GESTURE_FOLDER):

        raise FileNotFoundError(
            "gestures folder not found."
        )

    total_flipped = 0

    # LOOP THROUGH GESTURE IDS
    for g_id in os.listdir(GESTURE_FOLDER):

        gesture_path = os.path.join(
            GESTURE_FOLDER,
            g_id
        )

        # SKIP NON-FOLDERS
        if not os.path.isdir(
            gesture_path
        ):
            continue

        print(
            f"\nProcessing Gesture ID: "
            f"{g_id}"
        )

        # GET ALL IMAGE FILES
        image_files = [

            f for f in os.listdir(
                gesture_path
            )

            if f.lower().endswith(
                SUPPORTED_EXTENSIONS
            )
        ]

        # SORT FILES
        image_files.sort()

        # CURRENT IMAGE COUNT
        current_count = len(image_files)

        if current_count == 0:

            print(
                f"No images found in "
                f"{gesture_path}"
            )

            continue

        # CREATE FLIPPED IMAGES
        for index, image_file in enumerate(image_files):

            image_path = os.path.join(
                gesture_path,
                image_file
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

            # FLIP IMAGE
            flipped_img = cv2.flip(
                img,
                1
            )

            # NEW IMAGE NAME
            new_image_number = (
                current_count
                + index
                + 1
            )

            new_image_name = (
                f"{new_image_number}.jpg"
            )

            new_image_path = os.path.join(
                gesture_path,
                new_image_name
            )

            # SAVE IMAGE
            cv2.imwrite(
                new_image_path,
                flipped_img
            )

            total_flipped += 1

            print(
                f"Saved: "
                f"{new_image_path}"
            )

    print(
        f"\nTotal flipped images created: "
        f"{total_flipped}"
    )

    print(
        "\nDataset augmentation completed."
    )


# =========================
# START
# =========================
if __name__ == "__main__":

    flip_images()