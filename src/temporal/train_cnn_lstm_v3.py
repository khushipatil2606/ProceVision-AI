import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "balanced_split"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "procedural_error_cnn_lstm_v3.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 8
CHANNELS = 3

BATCH_SIZE = 16
EPOCHS = 25

SEED = 42


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_name):

    split_dir = (
        DATA_DIR / split_name
    )

    sequences = []
    labels = []

    for class_name, label in [
        ("correct", 0),
        ("error", 1)
    ]:

        folder = (
            split_dir / class_name
        )

        files = sorted(
            folder.glob("*.npz")
        )

        print(
            f"Loading {split_name}/{class_name}: "
            f"{len(files)} sequences"
        )

        for file in files:

            data = np.load(
                file,
                allow_pickle=True
            )

            frames = data[
                "frames"
            ]

            if frames.shape != (
                SEQUENCE_LENGTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
                CHANNELS
            ):

                print(
                    f"Skipping invalid: "
                    f"{file.name}"
                )

                continue

            sequences.append(
                frames.astype(
                    np.float32
                )
            )

            labels.append(label)

    X = np.array(
        sequences,
        dtype=np.float32
    )

    y = np.array(
        labels,
        dtype=np.int32
    )

    # Shuffle training data
    if split_name == "train":

        rng = np.random.default_rng(
            SEED
        )

        indices = rng.permutation(
            len(X)
        )

        X = X[indices]
        y = y[indices]

    return X, y


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\n==========================================")
    print("BUILDING CNN-LSTM V3")
    print("==========================================")

    cnn = models.Sequential([

        layers.Input(
            shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                CHANNELS
            )
        ),

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.30
        )
    ])

    inputs = layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
            CHANNELS
        )
    )

    features = layers.TimeDistributed(
        cnn
    )(inputs)

    temporal = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=False
        )
    )(features)

    temporal = layers.Dropout(
        0.40
    )(temporal)

    dense = layers.Dense(
        64,
        activation="relu"
    )(temporal)

    dense = layers.Dropout(
        0.30
    )(dense)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(dense)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" CNN-LSTM V3 TRAINING")
    print("==========================================")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading training dataset..."
    )

    X_train, y_train = load_split(
        "train"
    )

    print(
        "\nLoading validation dataset..."
    )

    X_val, y_val = load_split(
        "validation"
    )

    print(
        "\nLoading test dataset..."
    )

    X_test, y_test = load_split(
        "test"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("DATASET SUMMARY")
    print("==========================================")

    print(
        f"Train      : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print(
        f"Test       : {X_test.shape}"
    )

    print("\nTRAIN")

    print(
        f"Correct : "
        f"{np.sum(y_train == 0)}"
    )

    print(
        f"Error   : "
        f"{np.sum(y_train == 1)}"
    )

    print("\nVALIDATION")

    print(
        f"Correct : "
        f"{np.sum(y_val == 0)}"
    )

    print(
        f"Error   : "
        f"{np.sum(y_val == 1)}"
    )

    print("\nTEST")

    print(
        f"Correct : "
        f"{np.sum(y_test == 0)}"
    )

    print(
        f"Error   : "
        f"{np.sum(y_test == 1)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    print("\n==========================================")
    print("MODEL SUMMARY")
    print("==========================================")

    model.summary()

    # --------------------------------------------------------
    # Directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_recall",
            mode="max",
            save_best_only=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n==========================================")
    print("STARTING CNN-LSTM V3 TRAINING")
    print("==========================================")

    model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\n==========================================")
    print("FINAL TEST")
    print("==========================================")

    results = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    for name, value in zip(
        model.metrics_names,
        results
    ):

        print(
            f"{name:<12}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print("\n==========================================")
    print("CNN-LSTM V3 TRAINING COMPLETE")
    print("==========================================")

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()