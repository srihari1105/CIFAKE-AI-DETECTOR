import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

from dataset.data_loader import get_datasets


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "cifake_custom_cnn_best.keras"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "model",
    "evaluation_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("CIFAKE MODEL EVALUATION")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# GPU
# ============================================================

print("\nGPU devices:")

for gpu in tf.config.list_physical_devices("GPU"):
    print(gpu)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_ds, val_ds, test_ds = get_datasets()

print("Datasets loaded.")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating predictions on official test set...")

y_true = []
y_prob = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    ).ravel()

    y_true.extend(
        labels.numpy().astype(int)
    )

    y_prob.extend(predictions)


y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Convert probabilities to class predictions
y_pred = (y_prob >= 0.5).astype(int)


print("\nPrediction generation completed.")

print("Number of samples:", len(y_true))


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 60)
print("TEST SET METRICS")
print("=" * 60)

print(f"Accuracy :  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")
print(f"ROC-AUC  :  {auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "REAL",
        "FAKE"
    ],
    digits=4,
    zero_division=0
)

print(report)


# Save report
with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(
        "CIFAKE MODEL CLASSIFICATION REPORT\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC  : {auc:.4f}\n\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nMatrix format:")
print("[[REAL correctly classified, REAL predicted FAKE]")
print(" [FAKE predicted REAL,      FAKE correctly classified]]")


fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "REAL",
        "FAKE"
    ]
)

display.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "CIFAKE - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "CIFAKE - ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

real_probs = y_prob[y_true == 0]
fake_probs = y_prob[y_true == 1]

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    real_probs,
    bins=50,
    alpha=0.6,
    label="REAL"
)

plt.hist(
    fake_probs,
    bins=50,
    alpha=0.6,
    label="FAKE"
)

plt.xlabel(
    "Predicted Probability of FAKE"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Prediction Probability Distribution"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "prediction_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "true_label": y_true,
    "predicted_probability": y_prob,
    "predicted_label": y_pred
})

prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nSaved files:")

for filename in sorted(
    os.listdir(RESULTS_DIR)
):

    print(
        os.path.join(
            RESULTS_DIR,
            filename
        )
    )

print("\nFinal Test Results:")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nDONE.")