import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from dataset.data_loader import get_datasets


train_ds, _, _ = get_datasets()

images, labels = next(iter(train_ds))

print("=" * 60)
print("DATA INSPECTION")
print("=" * 60)

print("Image shape:", images.shape)
print("Labels:", labels.numpy())

print("\nREAL count:", int(tf.reduce_sum(tf.cast(labels == 0, tf.int32))))
print("FAKE count:", int(tf.reduce_sum(tf.cast(labels == 1, tf.int32))))

print("\nOverall mean:", float(tf.reduce_mean(images)))
print("Overall std :", float(tf.math.reduce_std(images)))

# Compare pixel statistics between classes
real_images = tf.boolean_mask(images, labels == 0)
fake_images = tf.boolean_mask(images, labels == 1)

print("\nREAL mean:", float(tf.reduce_mean(real_images)))
print("REAL std :", float(tf.math.reduce_std(real_images)))

print("\nFAKE mean:", float(tf.reduce_mean(fake_images)))
print("FAKE std :", float(tf.math.reduce_std(fake_images)))

print("\nFirst REAL image mean:",
      float(tf.reduce_mean(real_images[0])) if len(real_images) > 0 else "NONE")

print("First FAKE image mean:",
      float(tf.reduce_mean(fake_images[0])) if len(fake_images) > 0 else "NONE")

print("=" * 60)