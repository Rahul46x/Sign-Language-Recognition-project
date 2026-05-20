import cv2
import pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
import pyttsx3

from tensorflow.keras.models import load_model
from threading import Thread

# =========================
# SETTINGS
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

engine = pyttsx3.init()
engine.setProperty('rate', 150)

is_voice_on = True

x, y, w, h = 300, 100, 300, 300


# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "cnn_model_keras2.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found."
    )

model = load_model(MODEL_PATH)


# =========================
# LOAD HISTOGRAM
# =========================
def get_hand_hist():

    if not os.path.exists("hist"):
        raise FileNotFoundError(
            "hist file not found."
        )

    with open("hist", "rb") as f:
        hist = pickle.load(f)

    return hist


hist = get_hand_hist()


# =========================
# IMAGE SIZE
# =========================
def get_image_size():

    img = cv2.imread(
        'gestures/0/100.jpg',
        0
    )

    if img is None:
        raise FileNotFoundError(
            "Dataset image not found."
        )

    return img.shape


image_x, image_y = get_image_size()


# =========================
# PROCESS IMAGE
# =========================
def keras_process_image(img):

    img = cv2.resize(
        img,
        (image_x, image_y)
    )

    img = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    img = np.reshape(
        img,
        (1, image_x, image_y, 1)
    )

    return img


# =========================
# PREDICT
# =========================
def keras_predict(model, image):

    processed = keras_process_image(image)

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    pred_class = np.argmax(prediction)

    pred_probability = np.max(prediction)

    return pred_probability, pred_class


# =========================
# GET TEXT FROM DB
# =========================
def get_pred_text_from_db(pred_class):

    conn = sqlite3.connect(
        "gesture_db.db"
    )

    cursor = conn.execute(
        "SELECT g_name FROM gesture WHERE g_id=?",
        (int(pred_class),)
    )

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return ""


# =========================
# GET PREDICTION
# =========================
def get_pred_from_contour(contour, thresh):

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

    pred_probab, pred_class = keras_predict(
        model,
        save_img
    )

    if pred_probab * 100 > 70:
        return get_pred_text_from_db(
            pred_class
        )

    return ""


# =========================
# OPERATOR
# =========================
def get_operator(pred_text):

    operators = {
        "1": "+",
        "2": "-",
        "3": "*",
        "4": "/",
        "5": "%",
        "6": "**",
        "7": ">>",
        "8": "<<",
        "9": "&",
        "0": "|"
    }

    return operators.get(pred_text, "")


# =========================
# SPEAK
# =========================
def say_text(text):

    if not is_voice_on:
        return

    while engine._inLoop:
        pass

    engine.say(text)
    engine.runAndWait()


# =========================
# CAMERA
# =========================
def initialize_camera():

    for index in range(3):

        cam = cv2.VideoCapture(index)

        if cam.isOpened():
            print(
                f"Camera opened at index {index}"
            )
            return cam

    raise Exception(
        "Unable to open camera."
    )


# =========================
# IMAGE PROCESSING
# =========================
def get_img_contour_thresh(img):

    img = cv2.flip(img, 1)

    imgHSV = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    dst = cv2.calcBackProject(
        [imgHSV],
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

    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    return img, contours, thresh


# =========================
# TEXT MODE
# =========================
def text_mode(cam):

    global is_voice_on

    text = ""
    word = ""

    count_same_frame = 0

    while True:

        ret, img = cam.read()

        if not ret:
            continue

        img = cv2.resize(
            img,
            (640, 480)
        )

        img, contours, thresh = \
            get_img_contour_thresh(img)

        old_text = text

        if len(contours) > 0:

            contour = max(
                contours,
                key=cv2.contourArea
            )

            if cv2.contourArea(contour) > 10000:

                text = get_pred_from_contour(
                    contour,
                    thresh
                )

                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 20:

                    if len(text) == 1:

                        Thread(
                            target=say_text,
                            args=(text,)
                        ).start()

                    word += text

                    count_same_frame = 0

        # BLACKBOARD
        blackboard = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        cv2.putText(
            blackboard,
            "TEXT MODE",
            (180, 50),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.5,
            (255, 0, 0),
            2
        )

        cv2.putText(
            blackboard,
            "Prediction: " + text,
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            blackboard,
            "Output: " + word,
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            blackboard,
            "Press C = Calculator",
            (30, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        result = np.hstack(
            (img, blackboard)
        )

        cv2.imshow(
            "Gesture Recognition",
            result
        )

        cv2.imshow(
            "Threshold",
            thresh
        )

        keypress = cv2.waitKey(1)

        if keypress == ord('q'):
            return 0

        elif keypress == ord('c'):
            return 2

        elif keypress == ord('v'):
            is_voice_on = not is_voice_on


# =========================
# CALCULATOR MODE
# =========================
def calculator_mode(cam):

    calc_text = ""

    while True:

        ret, img = cam.read()

        if not ret:
            continue

        img = cv2.resize(
            img,
            (640, 480)
        )

        img, contours, thresh = \
            get_img_contour_thresh(img)

        pred_text = ""

        if len(contours) > 0:

            contour = max(
                contours,
                key=cv2.contourArea
            )

            if cv2.contourArea(contour) > 10000:

                pred_text = get_pred_from_contour(
                    contour,
                    thresh
                )

                if pred_text != "":
                    calc_text += pred_text

        blackboard = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        cv2.putText(
            blackboard,
            "CALCULATOR MODE",
            (100, 50),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.5,
            (255, 0, 0),
            2
        )

        cv2.putText(
            blackboard,
            "Prediction: " + pred_text,
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            blackboard,
            calc_text,
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            blackboard,
            "Press T = Text Mode",
            (30, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        result = np.hstack(
            (img, blackboard)
        )

        cv2.imshow(
            "Gesture Recognition",
            result
        )

        cv2.imshow(
            "Threshold",
            thresh
        )

        keypress = cv2.waitKey(1)

        if keypress == ord('q'):
            return 0

        elif keypress == ord('t'):
            return 1

        elif keypress == ord('e'):

            try:
                result = eval(calc_text)

                calc_text = (
                    calc_text
                    + " = "
                    + str(result)
                )

                Thread(
                    target=say_text,
                    args=(str(result),)
                ).start()

            except:
                calc_text = "Invalid Expression"

        elif keypress == ord('c'):
            calc_text = ""


# =========================
# MAIN LOOP
# =========================
def recognize():

    cam = initialize_camera()

    mode = 1

    while True:

        if mode == 1:
            mode = text_mode(cam)

        elif mode == 2:
            mode = calculator_mode(cam)

        else:
            break

    cam.release()

    cv2.destroyAllWindows()


# =========================
# START
# =========================
if __name__ == "__main__":

    keras_predict(
        model,
        np.zeros(
            (50, 50),
            dtype=np.uint8
        )
    )

    recognize()