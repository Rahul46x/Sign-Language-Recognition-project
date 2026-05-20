import cv2
import numpy as np
import pickle
import os


# =========================
# SETTINGS
# =========================
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

SQUARE_SIZE = 10
SQUARE_GAP = 10

START_X = 420
START_Y = 140

ROWS = 10
COLS = 5


# =========================
# BUILD SAMPLE SQUARES
# =========================
def build_squares(img):

    x = START_X
    y = START_Y

    crop = None

    for row in range(ROWS):

        row_crop = None

        for col in range(COLS):

            sample = img[
                y:y + SQUARE_SIZE,
                x:x + SQUARE_SIZE
            ]

            # DRAW RECTANGLE
            cv2.rectangle(
                img,
                (x, y),
                (
                    x + SQUARE_SIZE,
                    y + SQUARE_SIZE
                ),
                (0, 255, 0),
                1
            )

            # STACK HORIZONTALLY
            if row_crop is None:
                row_crop = sample
            else:
                row_crop = np.hstack(
                    (row_crop, sample)
                )

            x += (
                SQUARE_SIZE
                + SQUARE_GAP
            )

        # STACK VERTICALLY
        if crop is None:
            crop = row_crop
        else:
            crop = np.vstack(
                (crop, row_crop)
            )

        x = START_X

        y += (
            SQUARE_SIZE
            + SQUARE_GAP
        )

    return crop


# =========================
# INITIALIZE CAMERA
# =========================
def initialize_camera():

    for index in range(3):

        cam = cv2.VideoCapture(index)

        if cam.isOpened():

            print(
                f"Camera opened at "
                f"index {index}"
            )

            return cam

    raise Exception(
        "Unable to access camera."
    )


# =========================
# CREATE HAND HISTOGRAM
# =========================
def get_hand_hist():

    cam = initialize_camera()

    hist = None

    print("\nInstructions:")
    print("----------------------")
    print("1. Place hand in green boxes")
    print("2. Press 'c' to capture")
    print("3. Press 's' to save")
    print("4. Press 'q' to quit")
    print("----------------------\n")

    while True:

        ret, img = cam.read()

        if not ret:
            print(
                "Failed to capture frame."
            )
            continue

        # FLIP IMAGE
        img = cv2.flip(img, 1)

        # RESIZE
        img = cv2.resize(
            img,
            (
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            )
        )

        # BUILD SQUARES
        imgCrop = build_squares(img)

        # HSV IMAGE
        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        keypress = cv2.waitKey(1)

        # CAPTURE HISTOGRAM
        if keypress == ord('c'):

            hsvCrop = cv2.cvtColor(
                imgCrop,
                cv2.COLOR_BGR2HSV
            )

            hist = cv2.calcHist(
                [hsvCrop],
                [0, 1],
                None,
                [180, 256],
                [0, 180, 0, 256]
            )

            cv2.normalize(
                hist,
                hist,
                0,
                255,
                cv2.NORM_MINMAX
            )

            print(
                "\nHistogram captured."
            )

        # SAVE HISTOGRAM
        elif keypress == ord('s'):

            if hist is not None:

                with open("hist", "wb") as f:

                    pickle.dump(hist, f)

                print(
                    "\nHistogram saved "
                    "successfully."
                )

                break

            else:

                print(
                    "\nPlease capture "
                    "histogram first "
                    "using 'c'."
                )

        # QUIT
        elif keypress == ord('q'):

            print(
                "\nExited without saving."
            )

            break

        # SHOW THRESHOLD
        if hist is not None:

            dst = cv2.calcBackProject(
                [hsv],
                [0, 1],
                hist,
                [0, 180, 0, 256],
                1
            )

            disc = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (10, 10)
            )

            cv2.filter2D(
                dst,
                -1,
                disc,
                dst
            )

            blur = cv2.GaussianBlur(
                dst,
                (11, 11),
                0
            )

            blur = cv2.medianBlur(
                blur,
                15
            )

            thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU
            )[1]

            thresh = cv2.merge(
                (
                    thresh,
                    thresh,
                    thresh
                )
            )

            cv2.imshow(
                "Threshold",
                thresh
            )

        # DISPLAY MAIN WINDOW
        cv2.putText(
            img,
            "Press C = Capture",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            "Press S = Save",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            img,
            "Press Q = Quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Set Hand Histogram",
            img
        )

    # RELEASE CAMERA
    cam.release()

    cv2.destroyAllWindows()


# =========================
# START
# =========================
if __name__ == "__main__":

    get_hand_hist()