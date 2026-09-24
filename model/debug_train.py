import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

from dataset.data_loader import get_datasets
from model.build_model import build_model


print("=" * 60)
print("CUSTOM CNN - QUICK OVERFITTING TEST")
print("=" * 60)

print("\nLoading datasets...")

train_ds, val_ds, test_ds = get_datasets()

# Small subset for fast testing
small_train = train_ds.take(10)
small_val = val_ds.take(5)

print("Training batches: 10")
print("Validation batches: 5")

print("\nBuilding custom CNN...")

model = build_model()

model.summary()

print("\nStarting training...")
print("=" * 60)

history = model.fit(
    small_train,
    validation_data=small_val,
    epochs=10,
    verbose=1
)

print("\n" + "=" * 60)
print("QUICK TEST COMPLETED")
print("=" * 60)

print("\nFinal training accuracy:",
      history.history["accuracy"][-1])

print("Final validation accuracy:",
      history.history["val_accuracy"][-1])

print("Final training AUC:",
      history.history["auc"][-1])

print("Final validation AUC:",
      history.history["val_auc"][-1])