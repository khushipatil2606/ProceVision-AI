import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "annotated_split"
)

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"


# ============================================================
# MODEL OUTPUT
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "procedural_error_cnn_lstm.keras"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 8

IMAGE_SIZE = 128

CHANNELS = 3

BATCH_SIZE = 8

EPOCHS = 20

SEED = 42


# ============================================================
# LOAD NPZ FILES
# ============================================================

def load_dataset(folder):

    sequences = []
    labels = []

    files = sorted(
        folder.glob("*.npz")
    )

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

        label = int(
            data["label"]
        )

        # Safety check
        if frames.shape != (
            SEQUENCE_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
            CHANNELS
        ):

            print(
                f"WARNING: Skipping "
                f"{file.name}"
            )

            print(
                f"Shape: {frames.shape}"
            )

            continue

        sequences.append(
            frames.astype(
                np.float32
            )
        )

        labels.append(
            label
        )

    if len(sequences) == 0:

        raise RuntimeError(
            f"No valid sequences found in "
            f"{folder}"
        )

    return (
        np.array(
            sequences,
            dtype=np.float32
        ),
        np.array(
            labels,
            dtype=np.float32
        )
    )


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all_data():

    print("\n==========================================")
    print("LOADING TEMPORAL DATASET")
    print("==========================================")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_correct, train_correct_labels = (
        load_dataset(
            TRAIN_DIR / "correct"
        )
    )

    train_error, train_error_labels = (
        load_dataset(
            TRAIN_DIR / "error"
        )
    )

    X_train = np.concatenate(
        [
            train_correct,
            train_error
        ],
        axis=0
    )

    y_train = np.concatenate(
        [
            train_correct_labels,
            train_error_labels
        ],
        axis=0
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_correct, val_correct_labels = (
        load_dataset(
            VAL_DIR / "correct"
        )
    )

    val_error, val_error_labels = (
        load_dataset(
            VAL_DIR / "error"
        )
    )

    X_val = np.concatenate(
        [
            val_correct,
            val_error
        ],
        axis=0
    )

    y_val = np.concatenate(
        [
            val_correct_labels,
            val_error_labels
        ],
        axis=0
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_correct, test_correct_labels = (
        load_dataset(
            TEST_DIR / "correct"
        )
    )

    test_error, test_error_labels = (
        load_dataset(
            TEST_DIR / "error"
        )
    )

    X_test = np.concatenate(
        [
            test_correct,
            test_error
        ],
        axis=0
    )

    y_test = np.concatenate(
        [
            test_correct_labels,
            test_error_labels
        ],
        axis=0
    )


    # --------------------------------------------------------
    # SHUFFLE TRAINING DATA
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    train_indices = rng.permutation(
        len(X_train)
    )

    X_train = X_train[
        train_indices
    ]

    y_train = y_train[
        train_indices
    ]


    # --------------------------------------------------------
    # SHUFFLE VALIDATION
    # --------------------------------------------------------

    val_indices = rng.permutation(
        len(X_val)
    )

    X_val = X_val[
        val_indices
    ]

    y_val = y_val[
        val_indices
    ]


    # --------------------------------------------------------
    # SHUFFLE TEST
    # --------------------------------------------------------

    test_indices = rng.permutation(
        len(X_test)
    )

    X_test = X_test[
        test_indices
    ]

    y_test = y_test[
        test_indices
    ]


    # --------------------------------------------------------
    # PRINT SHAPES
    # --------------------------------------------------------

    print("\nDataset shapes:")

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"y_train : {y_train.shape}"
    )

    print(
        f"X_val   : {X_val.shape}"
    )

    print(
        f"y_val   : {y_val.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"y_test  : {y_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# BUILD CNN-LSTM
# ============================================================

def build_model():

    print("\n==========================================")
    print("BUILDING CNN-LSTM MODEL")
    print("==========================================")

    # --------------------------------------------------------
    # CNN FEATURE EXTRACTOR
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CNN + LSTM
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
            CHANNELS
        )
    )


    # Apply CNN to every frame
    frame_features = layers.TimeDistributed(
        cnn
    )(inputs)


    # Temporal reasoning
    temporal_features = layers.LSTM(
        128,
        return_sequences=False
    )(frame_features)


    # Regularization
    temporal_features = layers.Dropout(
        0.4
    )(temporal_features)


    # Classification
    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(temporal_features)


    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

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
    print(" CNN + LSTM TEMPORAL MODEL")
    print("==========================================")


    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_all_data()


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model()


    print("\n==========================================")
    print("MODEL SUMMARY")
    print("==========================================")

    model.summary()


    # ========================================================
    # MODEL DIRECTORY
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CALLBACKS
    # ========================================================

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


    # ========================================================
    # TRAIN
    # ========================================================

    print("\n==========================================")
    print("STARTING CNN-LSTM TRAINING")
    print("==========================================")

    history = model.fit(

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


    # ========================================================
    # TEST
    # ========================================================

    print("\n==========================================")
    print("TESTING CNN-LSTM MODEL")
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
        f"Test Loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE
    # ========================================================

    model.save(
        MODEL_PATH
    )


    print("\n==========================================")
    print("CNN-LSTM TRAINING COMPLETE")
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