import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "cifake_custom_cnn_best.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "model",
    "gradcam_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (32, 32)

# Last convolutional layer of our trained CNN
LAST_CONV_LAYER = "conv2d_5"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("CIFAKE GRAD-CAM")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

last_conv_layer = model.get_layer(
    LAST_CONV_LAYER
)

print(
    f"Using Grad-CAM layer: {last_conv_layer.name}"
)


# ============================================================
# GRAD-CAM MODEL
# ============================================================
#
# Instead of creating a Functional model using model.inputs
# during every prediction, we create the Grad-CAM model once.
#
# This also avoids the Keras input-structure warning.
# ============================================================

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    # Same image size used during training
    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Training uses raw 0-255 pixel values.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_array):

    prediction = model(
        image_array,
        training=False
    )

    probability = float(
        prediction.numpy()[0][0]
    )

    if probability >= 0.5:

        label = "FAKE"
        confidence = probability

    else:

        label = "REAL"
        confidence = 1.0 - probability

    return label, confidence, probability


# ============================================================
# GRAD-CAM HEATMAP
# ============================================================

def make_gradcam_heatmap(
    image_array
):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array,
            training=False
        )

        prediction = predictions[:, 0]

    # Gradients of prediction with respect
    # to the final convolutional feature maps
    grads = tape.gradient(
        prediction,
        conv_outputs
    )

    # Average gradients spatially
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Weight feature maps by importance
    weighted_features = (
        conv_outputs * pooled_grads
    )

    # Combine feature maps
    heatmap = tf.reduce_mean(
        weighted_features,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    save_visualization=True
):

    print("\n" + "-" * 60)
    print("Processing:")
    print(image_path)

    # Load image
    original_image, image_array = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    label, confidence, probability = predict_image(
        image_array
    )

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap = make_gradcam_heatmap(
        image_array
    )

    # Resize heatmap to original 32x32 size
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    heatmap = np.array(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    output_path = None

    if save_visualization:

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{base_name}_gradcam.png"
        )

        plt.figure(
            figsize=(12, 4)
        )

        # Original image
        plt.subplot(
            1, 3, 1
        )

        plt.imshow(
            original_image
        )

        plt.title(
            "Original Image"
        )

        plt.axis(
            "off"
        )

        # Heatmap
        plt.subplot(
            1, 3, 2
        )

        plt.imshow(
            heatmap,
            cmap="jet"
        )

        plt.title(
            "Grad-CAM Heatmap"
        )

        plt.axis(
            "off"
        )

        # Overlay
        plt.subplot(
            1, 3, 3
        )

        plt.imshow(
            original_image
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45
        )

        plt.title(
            f"{label} ({confidence * 100:.2f}%)"
        )

        plt.axis(
            "off"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "prediction": label,
        "confidence": confidence,
        "probability_fake": probability,
        "heatmap": heatmap,
        "output_path": output_path
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nGrad-CAM script ready."
    )

    print(
        "\nUsage:"
    )

    print(
        "python model/gradcam.py <image_path>"
    )

    if len(sys.argv) < 2:

        print(
            "\nERROR: Please provide an image path."
        )

        print(
            "\nExample:"
        )

        print(
            r'python model/gradcam.py "C:\path\to\image.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(
        image_path
    ):

        print(
            "\nERROR: Image not found:"
        )

        print(
            image_path
        )

        sys.exit(1)

    generate_gradcam(
        image_path
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "GRAD-CAM COMPLETED"
    )

    print(
        "=" * 60
    )