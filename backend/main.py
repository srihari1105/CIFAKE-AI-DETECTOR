import os
import io
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "cifake_custom_cnn_best.keras"
)

GRADCAM_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "static",
    "gradcam"
)

IMAGE_SIZE = (32, 32)

LAST_CONV_LAYER = "conv2d_5"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(GRADCAM_DIR, exist_ok=True)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CIFAKE AI Detector API",
    description="AI-generated image detection API with Grad-CAM explainability",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://cifake-ai-detector.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "backend", "static")),
    name="static"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("CIFAKE AI DETECTOR API")
print("=" * 60)

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print(
    "Available GPUs:",
    tf.config.list_physical_devices("GPU")
)


# ============================================================
# GRAD-CAM MODEL
# ============================================================

last_conv_layer = model.get_layer(LAST_CONV_LAYER)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes into the format
    expected by the trained CNN model.
    """

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # The model was trained using raw 0-255 pixel values.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# GRAD-CAM HEATMAP
# ============================================================

def make_gradcam_heatmap(image_array):
    """
    Generate Grad-CAM heatmap for the uploaded image.
    """

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array,
            training=False
        )

        probability_fake = predictions[:, 0]

    gradients = tape.gradient(
        probability_fake,
        conv_outputs
    )

    # Average gradients across spatial dimensions
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weight feature maps using averaged gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam_image(
    original_image,
    heatmap,
    output_path
):
    """
    Create a visual Grad-CAM overlay.
    """

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Resize heatmap to original image size
    heatmap_image = Image.fromarray(
        heatmap_uint8
    ).resize(
        original_image.size
    )

    heatmap_array = np.array(
        heatmap_image
    )

    # Simple red heatmap
    heatmap_rgb = np.zeros(
        (
            heatmap_array.shape[0],
            heatmap_array.shape[1],
            3
        ),
        dtype=np.uint8
    )

    heatmap_rgb[:, :, 0] = heatmap_array

    # Original image
    original_array = np.array(
        original_image,
        dtype=np.float32
    )

    heatmap_rgb = heatmap_rgb.astype(
        np.float32
    )

    # Overlay
    overlay = (
        0.65 * original_array
        +
        0.35 * heatmap_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    result = Image.fromarray(
        overlay
    )

    result.save(
        output_path
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "CIFAKE AI Detector API is running",
        "status": "success"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": len(
            tf.config.list_physical_devices("GPU")
        ) > 0
    }


# ============================================================
# PREDICTION + GRAD-CAM ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = await file.read()

    try:
        original_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    image = original_image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model(
        image_array,
        training=False
    )

    probability_fake = float(
        prediction.numpy()[0][0]
    )
    print("================================")
    print("MODEL RAW OUTPUT:", probability_fake)
    print("================================")


    if probability_fake >= 0.5:

        label = "FAKE"
        confidence = probability_fake

    else:

        label = "REAL"
        confidence = 1.0 - probability_fake


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap = make_gradcam_heatmap(
        image_array
    )


    # --------------------------------------------------------
    # Save Grad-CAM image
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    gradcam_filename = (
        f"{unique_id}_gradcam.jpg"
    )

    gradcam_path = os.path.join(
        GRADCAM_DIR,
        gradcam_filename
    )

    create_gradcam_image(
        original_image,
        heatmap,
        gradcam_path
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "filename": file.filename,

        "prediction": label,

        "confidence": round(
            confidence,
            4
        ),

        "probability_fake": round(
            probability_fake,
            4
        ),

        "gradcam_url": (
            f"/static/gradcam/"
            f"{gradcam_filename}"
        )
    }