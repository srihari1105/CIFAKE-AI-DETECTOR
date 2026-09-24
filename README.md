# CIFAKE AI Detector

An end-to-end AI-generated image detection application built with a
custom TensorFlow/Keras CNN, FastAPI, React, Vite, and Grad-CAM
explainability.

> **Status:** End-to-end MVP is working and ready for portfolio/demo
> deployment. The current model is evaluated on the CIFAKE distribution;
> it is not a universal detector for every AI image generator.

## 1. Overview

The application:

1.  Accepts an uploaded image.
2.  Converts it to RGB and resizes it to `32x32`.
3.  Runs the image through a custom CNN.
4.  Predicts `REAL` or `FAKE`.
5.  Returns confidence and fake probability.
6.  Generates a Grad-CAM visualization.
7.  Displays the prediction and explanation in a React web UI.

### Stack

-   Python 3.11
-   TensorFlow/Keras 2.21
-   CUDA + cuDNN
-   NumPy, Pandas, Pillow, OpenCV
-   Scikit-learn, Matplotlib
-   FastAPI + Uvicorn
-   React + Vite
-   JavaScript / CSS
-   Grad-CAM
-   Git/GitHub
-   WSL2 + NVIDIA GPU

------------------------------------------------------------------------

## 2. Architecture

``` text
                    USER
                      |
                      v
             +------------------+
             |  React Frontend  |
             |      Vite        |
             +--------+---------+
                      |
                POST /predict
                      |
                      v
             +------------------+
             | FastAPI Backend  |
             +--------+---------+
                      |
              +-------+-------+
              |               |
              v               v
       Image Preprocess   Custom CNN
              |               |
              |          Fake Probability
              |               |
              +-------+-------+
                      |
                      v
                  Prediction
                      |
                      v
                   Grad-CAM
                      |
                      v
             Explainability Image
                      |
                      v
             React Result Screen
```

------------------------------------------------------------------------

## 3. ML Pipeline

``` text
CIFAKE Dataset
      |
      v
Dataset Verification
      |
      v
Train / Validation / Test CSVs
      |
      v
Image Loading + Resize
      |
      v
Custom CNN Training
      |
      v
Best Checkpoint
      |
      v
Official Test Evaluation
      |
      +--> Accuracy
      +--> Precision
      +--> Recall
      +--> F1
      +--> ROC-AUC
      +--> Confusion Matrix
      +--> ROC Curve
      |
      v
Grad-CAM
      |
      v
FastAPI
      |
      v
React Frontend
```

------------------------------------------------------------------------

## 4. Dataset

The project uses the **CIFAKE: Real and AI-Generated Synthetic Images**
dataset.

Original dataset:

-   60,000 REAL images
-   60,000 AI-generated images
-   100,000 training images
-   20,000 official test images

Classes:

``` text
REAL -> CIFAR-10-derived real images
FAKE -> Stable Diffusion v1.4 generated images
```

Project split:

``` text
Training   : 90,000 (45,000 REAL + 45,000 FAKE)
Validation : 10,000 (5,000 REAL + 5,000 FAKE)
Test       : 20,000 (10,000 REAL + 10,000 FAKE)
```

Dataset verification found:

``` text
Train/validation overlap : 0
Missing files             : 0
Class counts              : Correct
Official test             : Preserved
```

CSV files:

``` text
dataset/splits/
├── train.csv
├── validation.csv
└── test.csv
```

The dataset itself should not be committed to GitHub because of its
size.

------------------------------------------------------------------------

## 5. Preprocessing

The current model uses the native CIFAKE resolution:

``` text
32 x 32 x 3
```

Pipeline:

``` text
Image
  -> decode
  -> RGB
  -> resize 32x32
  -> CNN
```

The model was trained with raw `0-255` pixel values, so inference uses
the same preprocessing.

------------------------------------------------------------------------

## 6. CNN Architecture

``` text
Input: 32x32x3
       |
Conv2D 32
BatchNorm
Conv2D 32
MaxPooling
Dropout 0.15
       |
Conv2D 64
BatchNorm
Conv2D 64
MaxPooling
Dropout 0.20
       |
Conv2D 128
BatchNorm
Conv2D 128
MaxPooling
Dropout 0.25
       |
GlobalAveragePooling
       |
Dense 128
Dropout 0.40
       |
Dense 1 + Sigmoid
       |
FAKE Probability
```

Total parameters:

``` text
304,545
```

Training configuration:

``` text
Optimizer : Adam
Learning rate : 0.001
Loss : Binary Cross-Entropy
Metrics : Accuracy, ROC-AUC
```

Training also used model checkpointing, early stopping, and
learning-rate reduction.

Best model:

``` text
model/cifake_custom_cnn_best.keras
```

------------------------------------------------------------------------

## 7. Evaluation

Official CIFAKE test results:

  Metric        Result
  ----------- --------
  Accuracy      96.24%
  Precision     97.65%
  Recall        94.75%
  F1 Score      96.18%
  ROC-AUC       99.46%

Confusion matrix:

``` text
                 Predicted
                 REAL    FAKE

Actual REAL      9772     228
Actual FAKE       525    9475
```

Evaluation artifacts:

``` text
model/evaluation_results/
├── classification_report.txt
├── confusion_matrix.png
├── prediction_distribution.png
├── roc_curve.png
└── test_predictions.csv
```

------------------------------------------------------------------------

## 8. Grad-CAM

Grad-CAM provides a visual explanation of the model prediction.

The implementation uses the final convolutional layer:

``` text
conv2d_5
```

It generates a heatmap and overlay showing image regions that
contributed to the prediction.

Manual usage:

``` bash
python model/gradcam.py <path_to_image>
```

Results are saved under:

``` text
model/gradcam_results/
```

The FastAPI `/predict` endpoint also generates a Grad-CAM image for each
uploaded image.

------------------------------------------------------------------------

## 9. Backend

Main file:

``` text
backend/main.py
```

Startup flow:

``` text
FastAPI starts
    |
Load model once
    |
Receive image
    |
Preprocess
    |
Predict
    |
Generate Grad-CAM
    |
Save Grad-CAM
    |
Return JSON
```

Endpoints:

### `GET /`

Basic API endpoint.

### `GET /health`

Health check.

### `POST /predict`

Accepts an uploaded image.

Example response:

``` json
{
  "filename": "example.jpg",
  "prediction": "FAKE",
  "confidence": 0.9821,
  "probability_fake": 0.9821,
  "gradcam_url": "/static/gradcam/example_gradcam.jpg"
}
```

Prediction logic:

``` python
probability_fake = float(prediction.numpy()[0][0])

if probability_fake >= 0.5:
    label = "FAKE"
    confidence = probability_fake
else:
    label = "REAL"
    confidence = 1.0 - probability_fake
```

------------------------------------------------------------------------

## 10. Frontend

Frontend structure:

``` text
frontend/
├── package.json
└── src/
    ├── App.jsx
    ├── App.css
    ├── index.css
    ├── main.jsx
    ├── assets/
    │   └── hero.png
    └── components/
        ├── ImageUploader.jsx
        ├── PredictionResult.jsx
        └── GradCAM.jsx
```

Flow:

``` text
Choose Image
     |
Preview
     |
Analyze
     |
POST /predict
     |
Receive JSON
     |
Display:
- Prediction
- Confidence
- Fake Probability
- Grad-CAM
```

------------------------------------------------------------------------

## 11. Project Structure

``` text
CIFAKE-AI-DETECTOR/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── static/
│       └── gradcam/
├── dataset/
│   ├── create_dataset_splits.py
│   ├── data_loader.py
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── model/
│   ├── build_model.py
│   ├── train.py
│   ├── debug_train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── cifake_custom_cnn_best.keras
│   ├── cifake_custom_cnn_final.keras
│   ├── evaluation_results/
│   └── gradcam_results/
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       ├── main.jsx
│       ├── assets/
│       └── components/
└── README.md
```

------------------------------------------------------------------------

## 12. Requirements

Recommended environment:

``` text
Python 3.11
Node.js + npm
Windows 11 + WSL2 (for the development GPU setup)
NVIDIA GPU + compatible CUDA/cuDNN environment (optional for CPU use)
```

Python packages:

``` bash
pip install "tensorflow[and-cuda]==2.21.0"
pip install pandas==2.2.3 numpy scikit-learn pillow opencv-python matplotlib
pip install fastapi uvicorn python-multipart
```

Frontend:

``` bash
cd frontend
npm install
```

------------------------------------------------------------------------

## 13. GPU Setup

Development used:

``` text
NVIDIA GeForce RTX 4050 Laptop GPU
~6 GB VRAM
WSL2 Ubuntu
Python 3.11
TensorFlow 2.21.0
CUDA + cuDNN
```

Verify TensorFlow GPU:

``` bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected:

``` text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

The backend was also verified to load the model on the RTX 4050 and use
cuDNN.

------------------------------------------------------------------------

## 14. Run Locally

### Clone

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CIFAKE-AI-DETECTOR
```

### Create and activate Python environment

``` bash
python3.11 -m venv wsl_venv
source wsl_venv/bin/activate
```

### Install backend packages

``` bash
pip install "tensorflow[and-cuda]==2.21.0"
pip install pandas==2.2.3 numpy scikit-learn pillow opencv-python matplotlib
pip install fastapi uvicorn python-multipart
```

### Configure dataset

Update the dataset path in:

``` text
dataset/data_loader.py
```

so it points to your local CIFAKE `DATASET` directory.

### Start backend

From the project root:

``` bash
source wsl_venv/bin/activate
uvicorn backend.main:app --reload
```

Backend:

``` text
http://localhost:8000
```

Swagger:

``` text
http://localhost:8000/docs
```

### Start frontend

Open another terminal:

``` bash
cd frontend
npm install
npm run dev
```

Frontend:

``` text
http://localhost:5173
```

Open the frontend, upload an image, and click **Analyze Image**.

------------------------------------------------------------------------

## 15. Training From Scratch

Create/recreate dataset splits:

``` bash
python dataset/create_dataset_splits.py
```

Train:

``` bash
python model/train.py
```

Evaluate:

``` bash
python model/evaluate.py
```

Run Grad-CAM manually:

``` bash
python model/gradcam.py <path_to_image>
```

------------------------------------------------------------------------

## 16. Deployment Notes

The application is working end-to-end and is suitable for deployment as
an MVP/demo.

Before production deployment, change development-only configuration:

### Frontend API URL

The current frontend uses the local backend URL:

``` text
http://localhost:8000
```

For deployment, replace it with the deployed API URL. Prefer a Vite
environment variable such as:

``` text
VITE_API_URL
```

### CORS

The backend currently allows the local Vite origins:

``` text
http://localhost:5173
http://127.0.0.1:5173
```

For production, allow only the actual deployed frontend domain.

### Model

The deployed backend needs access to:

``` text
model/cifake_custom_cnn_best.keras
```

### GPU

GPU inference is optional. If the hosting platform provides an NVIDIA
GPU, TensorFlow can use it with the appropriate CUDA/cuDNN environment.
CPU inference can also be used for a demo, but may be slower.

### Storage

The backend currently generates Grad-CAM images under:

``` text
backend/static/gradcam/
```

For production, add appropriate cleanup or persistent/object storage if
many requests are expected.

------------------------------------------------------------------------

## 17. Current Limitations

The model should **not** be presented as a universal AI detector.

The current training data is CIFAKE-based:

``` text
REAL -> CIFAR-10-derived images
FAKE -> Stable Diffusion v1.4 images
```

The model achieved excellent performance on the official CIFAKE test
distribution, but external AI images can be misclassified.

For example, an external AI-generated poster was classified as REAL with
a fake probability of approximately `0.0004`. This demonstrates that
high in-distribution accuracy does not guarantee generalization to
unseen generators, image styles, resolutions, or domains.

Therefore, the current system is best described as:

> A CIFAKE-trained AI-generated image classification system with
> Grad-CAM explainability.

not:

> A universal detector that can reliably identify every AI-generated
> image.

------------------------------------------------------------------------

## 18. Future Improvements

### Broader training data

Add real and AI-generated images from multiple sources and generators,
such as:

-   Real photographs
-   Real portraits
-   Real landscapes
-   Stable Diffusion
-   Midjourney
-   DALL-E
-   Flux
-   Other generators

### Higher resolution

Evaluate moving from:

``` text
32x32
```

to:

``` text
128x128
```

or:

``` text
224x224
```

to preserve more visual information.

### External testing

Create a separate test set that was never used during training.

### Unseen-generator testing

Train on some generators and test on a generator not seen during
training.

### Stronger models

Experiment with:

-   ResNet
-   EfficientNet
-   ConvNeXt
-   Transfer learning

### Production engineering

Add:

-   Environment variables
-   Docker
-   Production CORS
-   File-size/type validation
-   Rate limiting
-   Authentication if required
-   Better temporary-file cleanup
-   Cloud/GPU deployment

------------------------------------------------------------------------

## 19. What This Project Demonstrates

### Machine Learning

-   Dataset preparation
-   Data splitting
-   Image preprocessing
-   CNN design
-   Binary classification
-   Training and validation
-   Checkpointing
-   Early stopping
-   Learning-rate scheduling
-   Model evaluation
-   ROC-AUC
-   Confusion matrices

### Deep Learning

-   TensorFlow/Keras
-   GPU acceleration
-   CUDA
-   cuDNN
-   CNN feature extraction
-   Sigmoid classification
-   Grad-CAM

### Backend

-   FastAPI
-   REST APIs
-   Multipart image uploads
-   TensorFlow model serving
-   JSON APIs
-   Static file serving
-   CORS

### Frontend

-   React
-   Vite
-   Components
-   State management
-   File upload
-   REST API integration
-   Prediction visualization
-   Explainability visualization

### Engineering

-   WSL2
-   Virtual environments
-   Git/GitHub
-   Debugging
-   ML + backend + frontend integration
-   End-to-end application development

------------------------------------------------------------------------

## 20. Project Status

``` text
Dataset preparation       ✅
Dataset splitting         ✅
Data validation           ✅
Data loader               ✅
GPU configuration         ✅
CNN architecture          ✅
Model training            ✅
Best model selection      ✅
Official evaluation       ✅
Metrics                   ✅
Confusion matrix          ✅
ROC curve                 ✅
Grad-CAM                  ✅
FastAPI server            ✅
Model prediction API      ✅
Image upload API          ✅
GPU inference             ✅
Backend Grad-CAM          ✅
React frontend            ✅
Frontend ↔ Backend        ✅
Local end-to-end testing  ✅
GitHub preparation        ⏳
Deployment                ⏳
Portfolio/LinkedIn        ⏳
```

------------------------------------------------------------------------

## 21. Author

**Singupurapu Srihari**

B.Tech CSE (AI & ML)

Project: **CIFAKE AI Detector**

Built with:

``` text
TensorFlow + FastAPI + React
```

------------------------------------------------------------------------

## 22. Disclaimer

This project is intended for educational, research, and demonstration
purposes.

A prediction from the model should not be treated as definitive proof
that an image is authentic or AI-generated. Performance depends on the
training distribution and the similarity between new images and the data
used to train the model.
