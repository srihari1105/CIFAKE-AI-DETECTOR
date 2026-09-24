import { useState } from "react";

function ImageUploader({ onAnalyze, loading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleAnalyze = () => {
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }

    onAnalyze(selectedFile);
  };

  return (
    <div className="uploader-section">

      <div className="upload-box">

        {!preview ? (
          <>
            <div className="upload-icon">📁</div>

            <h2>Upload an Image</h2>

            <p>
              Select an image to check whether it is
              real or AI-generated.
            </p>

            <label className="file-button">
              Choose Image
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                hidden
              />
            </label>
          </>
        ) : (
          <>
            <img
              src={preview}
              alt="Selected preview"
              className="image-preview"
            />

            <p className="selected-file">
              {selectedFile.name}
            </p>

            <div className="upload-actions">

              <label className="secondary-button">
                Change Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  hidden
                />
              </label>

              <button
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze Image"}
              </button>

            </div>
          </>
        )}

      </div>

    </div>
  );
}

export default ImageUploader;