function PredictionResult({ result }) {
  if (!result) {
    return null;
  }

  const confidencePercent = (
    result.confidence * 100
  ).toFixed(2);

  const fakeProbabilityPercent = (
    result.probability_fake * 100
  ).toFixed(2);

  const isFake = result.prediction === "FAKE";

  return (
    <section className="result-section">

      <div className="result-header">
        <span className="result-label">
          ANALYSIS RESULT
        </span>

        <h2>
          The image is classified as
        </h2>

        <div
          className={`prediction-badge ${
            isFake ? "fake" : "real"
          }`}
        >
          {result.prediction}
        </div>
      </div>

      <div className="metrics">

        <div className="metric-card">
          <span>Confidence</span>
          <strong>{confidencePercent}%</strong>
        </div>

        <div className="metric-card">
          <span>Fake Probability</span>
          <strong>{fakeProbabilityPercent}%</strong>
        </div>

      </div>

    </section>
  );
}

export default PredictionResult;