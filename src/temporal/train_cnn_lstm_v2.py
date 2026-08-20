import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
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
    / "step_labeled_split"
)

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "procedural_error_cnn_lstm_v2.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 8
CHANNELS = 3

BATCH_SIZE = 8
EPOCHS = 20

SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_class_data(folder, label):

    files = sorted(
        folder.glob("*.npz")
    )

    sequences = []
    labels = []

    print(
        f"Loading {folder.name}: "
        f"{len(files)} sequences"
    )

    for file in files:

        data = np.load(
            file,
            allow_pickle=True
        )

        frames = data["frames"]

        if frames.shape != (
            SEQUENCE_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
            CHANNELS
        ):

            print(
                f"WARNING: Skipping {file.name}"
            )

            continue

        sequences.append(
            frames.astype(
                np.float32
            )
        )

        labels.append(label)

    return sequences, labels


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_dir):

    correct_sequences, correct_labels = (
        load_class_data(
            split_dir / "correct",
            0
        )
    )

    error_sequences, error_labels = (
        load_class_data(
            split_dir / "error",
            1
        )
    )

    X = np.array(
        correct_sequences + error_sequences,
        dtype=np.float32
    )

    y = np.array(
        correct_labels + error_labels,
        dtype=np.int32
    )

    # Shuffle
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
    print("BUILDING CNN-LSTM V2")
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
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
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

    temporal = layers.LSTM(
        128,
        return_sequences=False
    )(features)

    temporal = layers.Dropout(
        0.4
    )(temporal)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(temporal)

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
            "accuracy"
        ]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" CNN-LSTM V2 TRAINING")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    X_train, y_train = load_split(
        TRAIN_DIR
    )

    print("\nLoading validation dataset...")

    X_val, y_val = load_split(
        VAL_DIR
    )

    print("\nLoading test dataset...")

    X_test, y_test = load_split(
        TEST_DIR
    )

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    print("\n==========================================")
    print("DATASET SUMMARY")
    print("==========================================")

    print(
        f"Train : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print(
        f"Test : {X_test.shape}"
    )

    print(
        f"\nTrain correct : "
        f"{np.sum(y_train == 0)}"
    )

    print(
        f"Train error : "
        f"{np.sum(y_train == 1)}"
    )

    print(
        f"\nValidation correct : "
        f"{np.sum(y_val == 0)}"
    )

    print(
        f"Validation error : "
        f"{np.sum(y_val == 1)}"
    )

    print(
        f"\nTest correct : "
        f"{np.sum(y_test == 0)}"
    )

    print(
        f"Test error : "
        f"{np.sum(y_test == 1)}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight
        in zip(classes, weights)
    }

    print("\n==========================================")
    print("CLASS WEIGHTS")
    print("==========================================")

    print(
        class_weights
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model()

    print("\n==========================================")
    print("MODEL SUMMARY")
    print("==========================================")

    model.summary()

    # --------------------------------------------------------
    # MODEL DIRECTORY
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

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
    print("STARTING CNN-LSTM V2 TRAINING")
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

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n==========================================")
    print("TESTING CNN-LSTM V2")
    print("==========================================")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print("\n==========================================")
    print("FINAL RESULTS")
    print("==========================================")

    print(
        f"Test Loss     : {test_loss:.4f}"
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
    print("CNN-LSTM V2 TRAINING COMPLETE")
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