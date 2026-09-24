function GradCAM({ imageUrl }) {
  if (!imageUrl) {
    return null;
  }

  return (
    <section className="gradcam-section">

      <div className="section-heading">
        <span>MODEL EXPLAINABILITY</span>

        <h2>Grad-CAM Visualization</h2>

        <p>
          The highlighted regions show areas that
          contributed to the model's prediction.
        </p>
      </div>

      <div className="gradcam-card">

        <img
          src={imageUrl}
          alt="Grad-CAM visualization"
          className="gradcam-image"
        />

      </div>

    </section>
  );
}

export default GradCAM;