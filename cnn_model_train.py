import os
import cv2
import pickle
import numpy as np
from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =========================
# GET IMAGE SIZE
# =========================
def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)

    if img is None:
        raise FileNotFoundError(
            "Dataset image not found. "
            "Check gestures/1/100.jpg"
        )

    return img.shape


# =========================
# NUMBER OF CLASSES
# =========================
def get_num_of_classes():
    return len(glob('gestures/*'))


image_x, image_y = get_image_size()


# =========================
# CNN MODEL
# =========================
def cnn_model():

    num_of_classes = get_num_of_classes()

    model = Sequential([

        # BLOCK 1
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(image_x, image_y, 1)
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # BLOCK 2
        Conv2D(
            64,
            (3, 3),
            activation='relu'
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # BLOCK 3
        Conv2D(
            128,
            (3, 3),
            activation='relu'
        ),

        BatchNormalization(),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # FLATTEN
        Flatten(),

        # DENSE LAYER
        Dense(
            256,
            activation='relu'
        ),

        Dropout(0.5),

        # OUTPUT
        Dense(
            num_of_classes,
            activation='softmax'
        )
    ])

    # COMPILE MODEL
    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # SAVE BEST MODEL
    checkpoint = ModelCheckpoint(
        "cnn_model_keras2.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # EARLY STOPPING
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    return model, [checkpoint, early_stop]


# =========================
# LOAD DATA
# =========================
def load_data():

    # TRAIN IMAGES
    with open("train_images", "rb") as f:
        train_images = np.array(
            pickle.load(f)
        )

    # TRAIN LABELS
    with open("train_labels", "rb") as f:
        train_labels = np.array(
            pickle.load(f),
            dtype=np.int32
        )

    # VALIDATION IMAGES
    with open("val_images", "rb") as f:
        val_images = np.array(
            pickle.load(f)
        )

    # VALIDATION LABELS
    with open("val_labels", "rb") as f:
        val_labels = np.array(
            pickle.load(f),
            dtype=np.int32
        )

    # RESHAPE + NORMALIZE
    train_images = train_images.reshape(
        -1,
        image_x,
        image_y,
        1
    ).astype("float32") / 255.0

    val_images = val_images.reshape(
        -1,
        image_x,
        image_y,
        1
    ).astype("float32") / 255.0

    # ONE HOT ENCODING
    train_labels = to_categorical(
        train_labels
    )

    val_labels = to_categorical(
        val_labels
    )

    return (
        train_images,
        train_labels,
        val_images,
        val_labels
    )


# =========================
# TRAIN MODEL
# =========================
def train():

    (
        train_images,
        train_labels,
        val_images,
        val_labels
    ) = load_data()

    model, callbacks = cnn_model()

    # MODEL SUMMARY
    model.summary()

    # DATA AUGMENTATION
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    datagen.fit(train_images)

    # TRAIN MODEL
    model.fit(
        datagen.flow(
            train_images,
            train_labels,
            batch_size=32
        ),

        validation_data=(
            val_images,
            val_labels
        ),

        epochs=30,

        callbacks=callbacks
    )

    # EVALUATE MODEL
    loss, accuracy = model.evaluate(
        val_images,
        val_labels,
        verbose=0
    )

    print(
        f"\nValidation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # SAVE MODEL
    model.save("cnn_model_keras2.h5")

    print("\nModel saved successfully.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()