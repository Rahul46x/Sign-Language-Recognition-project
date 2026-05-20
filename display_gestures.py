import cv2
import os
import random
import numpy as np


# =========================
# GET IMAGE SIZE
# =========================
def get_image_size():

    img = cv2.imread(
        'gestures/0/100.jpg',
        0
    )

    if img is None:
        raise FileNotFoundError(
            "Sample image not found.\n"
            "Check gestures/0/100.jpg"
        )

    return img.shape


# =========================
# LOAD GESTURES
# =========================
if not os.path.exists("gestures"):
    raise FileNotFoundError(
        "gestures folder not found."
    )

gestures = os.listdir("gestures")

# KEEP ONLY NUMERIC FOLDERS
gestures = [
    g for g in gestures
    if g.isdigit()
]

# SORT NUMERICALLY
gestures.sort(key=int)

# IMAGE SIZE
image_x, image_y = get_image_size()

# GRID SETTINGS
images_per_row = 5

rows = (
    len(gestures) + images_per_row - 1
) // images_per_row

full_img = None


# =========================
# BUILD GRID
# =========================
for row in range(rows):

    row_images = []

    start_index = row * images_per_row
    end_index = min(
        start_index + images_per_row,
        len(gestures)
    )

    for i in range(start_index, end_index):

        gesture_id = gestures[i]

        gesture_folder = os.path.join(
            "gestures",
            gesture_id
        )

        # GET ALL IMAGES
        image_files = [
            f for f in os.listdir(gesture_folder)
            if f.endswith(".jpg")
        ]

        # HANDLE EMPTY FOLDER
        if len(image_files) == 0:

            img = np.zeros(
                (image_y, image_x),
                dtype=np.uint8
            )

        else:

            # RANDOM IMAGE
            random_image = random.choice(
                image_files
            )

            img_path = os.path.join(
                gesture_folder,
                random_image
            )

            img = cv2.imread(
                img_path,
                0
            )

            # HANDLE FAILED READ
            if img is None:

                img = np.zeros(
                    (image_y, image_x),
                    dtype=np.uint8
                )

        # ADD LABEL
        labeled_img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

        cv2.putText(
            labeled_img,
            f"ID: {gesture_id}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        row_images.append(labeled_img)

    # FILL EMPTY SPACES
    while len(row_images) < images_per_row:

        blank = np.zeros(
            (image_y, image_x, 3),
            dtype=np.uint8
        )

        row_images.append(blank)

    # STACK HORIZONTALLY
    row_img = np.hstack(row_images)

    # STACK VERTICALLY
    if full_img is None:
        full_img = row_img
    else:
        full_img = np.vstack(
            (full_img, row_img)
        )


# =========================
# SAVE OUTPUT
# =========================
output_path = "full_img.jpg"

cv2.imwrite(
    output_path,
    full_img
)

print(
    f"\nGesture preview saved as "
    f"{output_path}"
)


# =========================
# DISPLAY OUTPUT
# =========================
cv2.imshow(
    "Gesture Dataset Preview",
    full_img
)

print("\nPress any key to exit...")

cv2.waitKey(0)

cv2.destroyAllWindows()