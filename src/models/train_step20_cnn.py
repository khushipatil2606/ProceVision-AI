import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_split"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_split"
    / "validation"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_split"
    / "test"
)


# ============================================================
# MODEL OUTPUT
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "step20_cnn.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 8

EPOCHS = 20

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP 20 CNN")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=["correct", "error"],
        shuffle=True,
        seed=SEED
    )

    print("\nLoading validation dataset...")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=["correct", "error"],
        shuffle=False
    )

    print("\nLoading test dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=["correct", "error"],
        shuffle=False
    )

    print("\nClasses:")
    print(train_ds.class_names)

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        AUTOTUNE
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ])

    # --------------------------------------------------------
    # CNN MODEL
    # --------------------------------------------------------

    model = models.Sequential([

        layers.Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3
            )
        ),

        data_augmentation,

        layers.Rescaling(
            1.0 / 255
        ),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Dropout(0.30),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.40),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    print("\n==========================================")
    print("MODEL ARCHITECTURE")
    print("==========================================")

    model.summary()

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n==========================================")
    print("STARTING STEP 20 TRAINING")
    print("==========================================")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n==========================================")
    print("TESTING STEP 20 MODEL")
    print("==========================================")

    test_loss, test_accuracy = model.evaluate(
        test_ds
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print("\n==========================================")
    print("STEP 20 TRAINING COMPLETE")
    print("==========================================")

    print(
        f"Model saved to:\n{MODEL_PATH}"
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()