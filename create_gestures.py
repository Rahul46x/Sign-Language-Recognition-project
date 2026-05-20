import cv2
import numpy as np
import pickle
import os
import sqlite3
import random

# =========================
# IMAGE SIZE
# =========================
image_x, image_y = 50, 50


# =========================
# LOAD HAND HISTOGRAM
# =========================
def get_hand_hist():

    if not os.path.exists("hist"):
        raise FileNotFoundError(
            "hist file not found. "
            "Run set_hand_histogram.py first."
        )

    with open("hist", "rb") as f:
        hist = pickle.load(f)

    return hist


# =========================
# CREATE DATABASE + FOLDER
# =========================
def init_create_folder_database():

    # CREATE GESTURES FOLDER
    if not os.path.exists("gestures"):
        os.mkdir("gestures")

    # CREATE DATABASE
    if not os.path.exists("gesture_db.db"):

        conn = sqlite3.connect("gesture_db.db")

        create_table_cmd = """
        CREATE TABLE IF NOT EXISTS gesture (
            g_id INTEGER PRIMARY KEY,
            g_name TEXT NOT NULL
        )
        """

        conn.execute(create_table_cmd)

        conn.commit()
        conn.close()

        print("Database created successfully.")


# =========================
# CREATE FOLDER
# =========================
def create_folder(folder_name):

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


# =========================
# STORE IN DATABASE
# =========================
def store_in_db(g_id, g_name):

    conn = sqlite3.connect("gesture_db.db")

    try:

        conn.execute(
            "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)",
            (g_id, g_name)
        )

        print("Gesture stored successfully.")

    except sqlite3.IntegrityError:

        choice = input(
            "Gesture ID already exists. "
            "Update existing record? (y/n): "
        )

        if choice.lower() == 'y':

            conn.execute(
                "UPDATE gesture SET g_name=? WHERE g_id=?",
                (g_name, g_id)
            )

            print("Record updated successfully.")

        else:
            print("No changes made.")

    conn.commit()
    conn.close()


# =========================
# OPEN CAMERA
# =========================
def initialize_camera():

    cam = None

    for index in range(3):

        cam = cv2.VideoCapture(index)

        if cam.isOpened():
            print(f"Camera opened successfully at index {index}")
            return cam

    raise Exception("Unable to access camera.")


# =========================
# STORE IMAGES
# =========================
def store_images(g_id):

    total_pics = 1200

    hist = get_hand_hist()

    cam = initialize_camera()

    x, y, w, h = 300, 100, 300, 300

    create_folder("gestures/" + str(g_id))

    pic_no = 0

    flag_start_capturing = False

    frames = 0

    while True:

        ret, img = cam.read()

        if not ret:
            print("Failed to capture image.")
            continue

        img = cv2.flip(img, 1)

        imgHSV = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        # BACK PROJECTION
        dst = cv2.calcBackProject(
            [imgHSV],
            [0, 1],
            hist,
            [0, 180, 0, 256],
            1
        )

        # FILTERING
        disc = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (10, 10)
        )

        cv2.filter2D(dst, -1, disc, dst)

        blur = cv2.GaussianBlur(
            dst,
            (11, 11),
            0
        )

        blur = cv2.medianBlur(
            blur,
            15
        )

        # THRESHOLD
        thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        thresh = cv2.merge(
            (thresh, thresh, thresh)
        )

        thresh = cv2.cvtColor(
            thresh,
            cv2.COLOR_BGR2GRAY
        )

        thresh = thresh[
            y:y + h,
            x:x + w
        ]

        # FIND CONTOURS
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
        )

        # PROCESS CONTOUR
        if len(contours) > 0:

            contour = max(
                contours,
                key=cv2.contourArea
            )

            if (
                cv2.contourArea(contour) > 10000
                and frames > 50
            ):

                x1, y1, w1, h1 = cv2.boundingRect(
                    contour
                )

                save_img = thresh[
                    y1:y1 + h1,
                    x1:x1 + w1
                ]

                # MAKE IMAGE SQUARE
                if w1 > h1:

                    save_img = cv2.copyMakeBorder(
                        save_img,
                        int((w1 - h1) / 2),
                        int((w1 - h1) / 2),
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=0
                    )

                elif h1 > w1:

                    save_img = cv2.copyMakeBorder(
                        save_img,
                        0,
                        0,
                        int((h1 - w1) / 2),
                        int((h1 - w1) / 2),
                        cv2.BORDER_CONSTANT,
                        value=0
                    )

                # RESIZE IMAGE
                save_img = cv2.resize(
                    save_img,
                    (image_x, image_y)
                )

                # RANDOM FLIP
                if random.randint(0, 1):

                    save_img = cv2.flip(
                        save_img,
                        1
                    )

                pic_no += 1

                save_path = (
                    "gestures/"
                    + str(g_id)
                    + "/"
                    + str(pic_no)
                    + ".jpg"
                )

                cv2.imwrite(
                    save_path,
                    save_img
                )

                cv2.putText(
                    img,
                    "Capturing...",
                    (30, 60),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    2,
                    (127, 255, 255),
                    2
                )

        # DRAW RECTANGLE
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # SHOW IMAGE COUNT
        cv2.putText(
            img,
            f"Images: {pic_no}/{total_pics}",
            (30, 420),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (127, 127, 255),
            2
        )

        # SHOW WINDOWS
        cv2.imshow(
            "Capturing Gesture",
            img
        )

        cv2.imshow(
            "Threshold",
            thresh
        )

        keypress = cv2.waitKey(1)

        # START / STOP CAPTURE
        if keypress == ord('c'):

            flag_start_capturing = (
                not flag_start_capturing
            )

            if not flag_start_capturing:
                frames = 0

        # EXIT
        if keypress == ord('q'):
            print("Capture stopped by user.")
            break

        # COUNT FRAMES
        if flag_start_capturing:
            frames += 1

        # FINISH
        if pic_no >= total_pics:
            print("Image capture completed.")
            break

    # RELEASE CAMERA
    cam.release()

    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    init_create_folder_database()

    g_id = int(
        input("Enter gesture number: ")
    )

    g_name = input(
        "Enter gesture name/text: "
    )

    store_in_db(g_id, g_name)

    store_images(g_id)