from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "split_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "procedural_error_cnn.keras"


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15


# ==========================================
# LOAD DATASETS
# ==========================================

print("\n==========================================")
print("   PROCEVISION AI - CNN TRAINING")
print("==========================================")

print("\nLoading training data...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

print("\nLoading validation data...")

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nLoading test data...")

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


print("\nClasses:")
print(train_data.class_names)


# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)


# ==========================================
# CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\n==========================================")
print("MODEL ARCHITECTURE")
print("==========================================")

model.summary()


# ==========================================
# TRAIN
# ==========================================

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


# ==========================================
# TEST
# ==========================================

print("\n==========================================")
print("TESTING MODEL")
print("==========================================")

test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# ==========================================
# SAVE MODEL
# ==========================================

model.save(MODEL_PATH)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(f"Model saved to:")
print(MODEL_PATH)

print("==========================================\n")