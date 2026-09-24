import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

from dataset.data_loader import get_datasets
from model.build_model import build_model


MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cifake_custom_cnn_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cifake_custom_cnn_final.keras"
)


EPOCHS = 20


print("=" * 60)
print("CIFAKE CUSTOM CNN TRAINING")
print("=" * 60)

print("\nGPU devices:")

for gpu in tf.config.list_physical_devices("GPU"):
    print(gpu)


print("\nLoading datasets...")

train_ds, val_ds, test_ds = get_datasets()

print("\nDatasets loaded successfully.")


print("\nBuilding custom CNN...")

model = build_model()

model.summary()


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


print("\n" + "=" * 60)
print("STARTING FULL TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)


model.save(FINAL_MODEL_PATH)

print("\nBest model:")
print(BEST_MODEL_PATH)

print("\nFinal model:")
print(FINAL_MODEL_PATH)


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

val_results = model.evaluate(
    val_ds,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    val_results
):
    print(f"{name}: {value:.4f}")


print("\n" + "=" * 60)
print("OFFICIAL TEST EVALUATION")
print("=" * 60)

test_results = model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    test_results
):
    print(f"{name}: {value:.4f}")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)