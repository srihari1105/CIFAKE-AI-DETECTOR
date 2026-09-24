import { useState } from "react";

import ImageUploader from "./components/ImageUploader";
import PredictionResult from "./components/PredictionResult";
import GradCAM from "./components/GradCAM";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeImage = async (file) => {
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();

    formData.append("file", file);

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          "Failed to analyze the image."
        );
      }

      const data = await response.json();

      setResult(data);

    } catch (err) {
      console.error(err);

      setError(
        "Could not connect to the CIFAKE API. Make sure the FastAPI server is running."
      );

    } finally {
      setLoading(false);
    }
  };

  const gradcamUrl = result
    ? `${import.meta.env.VITE_API_URL}${result.gradcam_url}`
    : null;

  return (
    <div className="app">

      <header className="hero">

        <div className="hero-content">

          <span className="eyebrow">
            AI IMAGE FORENSICS
          </span>

          <h1>
            CIFAKE
            <br />
            <span>AI Detector</span>
          </h1>

          <p>
            Detect whether an image is real or
            AI-generated using a custom convolutional
            neural network with visual explainability.
          </p>

        </div>

      </header>


      <main>

        <ImageUploader
          onAnalyze={analyzeImage}
          loading={loading}
        />


        {error && (
          <div className="error-message">
            {error}
          </div>
        )}


        <PredictionResult
          result={result}
        />


        <GradCAM
          imageUrl={gradcamUrl}
        />

      </main>


      <footer>
        <p>
          CIFAKE AI Detector • TensorFlow • FastAPI • React
        </p>
      </footer>

    </div>
  );
}

export default App;