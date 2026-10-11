
import { useState } from "react";
import "./App.css";

const initialValues = {
  sepal_length: "5.1",
  sepal_width: "3.5",
  petal_length: "1.4",
  petal_width: "0.2",
};

const measurements = [
  { name: "sepal_length", label: "Sepal length", unit: "cm" },
  { name: "sepal_width", label: "Sepal width", unit: "cm" },
  { name: "petal_length", label: "Petal length", unit: "cm" },
  { name: "petal_width", label: "Petal width", unit: "cm" },
];

const flowerInfo = {
  "Iris-setosa": {
    title: "Iris Setosa",
    description: "A distinctive variety with small petals and a compact flower shape.",
    color: "setosa",
    symbol: "✿",
  },
  "Iris-versicolor": {
    title: "Iris Versicolor",
    description: "A balanced variety with medium-sized petals and sepals.",
    color: "versicolor",
    symbol: "❋",
  },
  "Iris-virginica": {
    title: "Iris Virginica",
    description: "A variety typically characterized by larger petals and sepals.",
    color: "virginica",
    symbol: "✾",
  },
};

function App() {
  const [values, setValues] = useState(initialValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleChange(event) {
    setValues({ ...values, [event.target.name]: event.target.value });
  }

  async function handlePredict(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(values).map(([key, value]) => [key, Number(value)])
          )
        ),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to make a prediction.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the prediction API. Make sure your FastAPI server is running at http://127.0.0.1:8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setValues(initialValues);
    setResult(null);
    setError("");
  }

  const flower = result ? flowerInfo[result.predicted_species] : null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home">
          <span className="brand-mark">✿</span>
          <span>Flora<span className="brand-light">AI</span></span>
        </a>
        <div className="topbar-right">
          <span className="status-dot" />
          <span>Neural network prediction</span>
          <span className="version-tag">v1.0</span>
        </div>
      </header>

      <main id="home">
        <section className="hero">
          <div className="eyebrow">
            <span>✦</span> MACHINE LEARNING · IRIS CLASSIFICATION
          </div>
          <h1>Every flower has<br />a <span>story to tell.</span></h1>
          <p className="hero-description">
            Discover the species of an Iris flower using four simple
            measurements and the power of a trained neural network.
          </p>
          <div className="hero-stats">
            <div><strong>3</strong><span>Flower species</span></div>
            <div className="stat-divider" />
            <div><strong>4</strong><span>Measurements</span></div>
            <div className="stat-divider" />
            <div><strong>93.33%</strong><span>Test accuracy</span></div>
          </div>
        </section>

        <section className="workspace">
          <div className="form-card">
            <div className="card-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Flower measurements</h2>
                <p>Enter the dimensions of your Iris flower.</p>
              </div>
              <div className="ruler-icon">⌁</div>
            </div>

            <form onSubmit={handlePredict}>
              <div className="measurement-grid">
                {measurements.map((measurement, index) => (
                  <label className="measurement-field" key={measurement.name}>
                    <span className="field-label">
                      <span className="field-number">0{index + 1}</span>
                      {measurement.label}
                    </span>
                    <span className="input-wrap">
                      <input
                        type="number"
                        name={measurement.name}
                        value={values[measurement.name]}
                        onChange={handleChange}
                        min="0.1"
                        max="10"
                        step="0.1"
                        required
                        aria-label={measurement.label}
                      />
                      <span className="unit">{measurement.unit}</span>
                    </span>
                  </label>
                ))}
              </div>

              <div className="form-note">
                <span>ⓘ</span>
                Measurements should be in centimetres (cm).
              </div>

              <button className="predict-button" type="submit" disabled={loading}>
                {loading ? "Analyzing flower..." : "Predict flower species"}
                {!loading && <span>↗</span>}
              </button>
              <button className="reset-button" type="button" onClick={resetForm}>
                Reset measurements
              </button>
            </form>
          </div>

          <div className={`result-card ${result ? "has-result" : ""}`}>
            <div className="card-heading result-heading">
              <div>
                <span className="step-label">STEP 02</span>
                <h2>Prediction result</h2>
                <p>Your AI-powered classification.</p>
              </div>
              <div className="result-icon">✳</div>
            </div>

            {error ? (
              <div className="error-panel">
                <span className="error-symbol">!</span>
                <h3>Prediction unavailable</h3>
                <p>{error}</p>
              </div>
            ) : result && flower ? (
              <div className="prediction-content">
                <div className={`flower-emblem ${flower.color}`}>
                  <span>{flower.symbol}</span>
                </div>
                <span className="result-caption">PREDICTED SPECIES</span>
                <h3 className="species-name">{flower.title}</h3>
                <p className="species-description">{flower.description}</p>

                <div className="confidence-panel">
                  <div className="confidence-top">
                    <span>Model confidence</span>
                    <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="confidence-track">
                    <div
                      className={`confidence-fill ${flower.color}`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                <div className="probability-section">
                  <h4>Species probability</h4>
                  {Object.entries(result.probabilities).map(([species, probability]) => (
                    <div className="probability-row" key={species}>
                      <div className="probability-label">
                        <span>{flowerInfo[species]?.title || species}</span>
                        <strong>{(probability * 100).toFixed(2)}%</strong>
                      </div>
                      <div className="probability-track">
                        <div
                          className={`probability-fill ${flowerInfo[species]?.color || ""}`}
                          style={{ width: `${probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="empty-result">
                <div className="empty-illustration">
                  <span className="leaf leaf-one">❧</span>
                  <span className="empty-flower">✿</span>
                  <span className="leaf leaf-two">❧</span>
                </div>
                <h3>Your result will bloom here.</h3>
                <p>Enter the four measurements and run a prediction to discover the flower species.</p>
                <div className="empty-badge"><span /> READY FOR ANALYSIS</div>
              </div>
            )}

            <div className="result-footer">
              <span>✧</span> Powered by your trained neural network
            </div>
          </div>
        </section>

        <section className="species-section">
          <div className="section-intro">
            <span className="step-label">GET TO KNOW THE FLOWERS</span>
            <h2>Three species. One intelligent model.</h2>
            <p>Your model distinguishes between these three Iris varieties.</p>
          </div>
          <div className="species-grid">
            {Object.entries(flowerInfo).map(([key, info]) => (
              <article className="species-card" key={key}>
                <div className={`species-symbol ${info.color}`}>{info.symbol}</div>
                <div>
                  <h3>{info.title}</h3>
                  <p>{info.description}</p>
                </div>
                <span className="species-arrow">↗</span>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand footer-brand" href="#home">
          <span className="brand-mark">✿</span>
          <span>Flora<span className="brand-light">AI</span></span>
        </a>
        <span>Built with React, FastAPI & machine learning.</span>
        <span className="footer-note">A learning project in AI</span>
      </footer>
    </div>
  );
}

export default App;
