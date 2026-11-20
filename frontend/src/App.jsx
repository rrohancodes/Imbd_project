import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [formData, setFormData] = useState({
    Meta_score: "",
    No_of_Votes: "",
    Gross: "",
    Runtime: "",
    Released_Year: "",
    Genre: "",
    Certificate: "",
    Director: "",
    Overview: ""
  });

  const [movieName, setMovieName] = useState("");

const [history, setHistory] = useState(() => {
  return JSON.parse(localStorage.getItem("predictionHistory")) || [];
});

const [actualRating, setActualRating] = useState("");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleChange(e) {
    const { name, value } = e.target;

    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = {
        ...formData,
        Meta_score: Number(formData.Meta_score),
        No_of_Votes: Number(formData.No_of_Votes),
        Gross: Number(formData.Gross),
        Runtime: Number(formData.Runtime),
        Released_Year: Number(formData.Released_Year)
      };

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        payload
      );

      setResult(response.data);
      savePrediction(response.data);

    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Something went wrong. Please check the backend server."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
  localStorage.setItem("predictionHistory", JSON.stringify(history));
}, [history]);

function savePrediction(data) {
  const newEntry = {
    id: Date.now(),
    movieName: movieName || "Untitled Movie",
    predictedRating: data.predicted_rating,
    category: data.predicted_category,
    actualRating: null,
    error: null,
    date: new Date().toLocaleDateString()
  };

  setHistory(prev => [newEntry, ...prev]);
}

function deletePrediction(id) {
  setHistory(prev => prev.filter(movie => movie.id !== id));
}

function updateActualRating(id) {
  if (actualRating === "" || Number(actualRating) < 0 || Number(actualRating) > 10) {
    return;
  }

  setHistory(prev =>
    prev.map(movie =>
      movie.id === id
        ? {
            ...movie,
            actualRating: Number(actualRating),
            error: Math.abs(
              movie.predictedRating - Number(actualRating)
            ).toFixed(2)
          }
        : movie
    )
  );

  setActualRating("");
}

  return (
    <div className="app">

      <header>
        <h1>IMDb Predictor</h1>
        <p>Discover the potential of your next movie.</p>
      </header>

      <main>

        <form onSubmit={handleSubmit}>

          <h2>Movie Information</h2>

          <div className="form-grid">

            <input
  type="text"
  placeholder="Movie Name"
  value={movieName}
  onChange={e => setMovieName(e.target.value)}
/>

            <input
              type="number"
              name="Meta_score"
              placeholder="Metascore (0–100)"
              min="0"
              max="100"
              step="any"
              value={formData.Meta_score}
              onChange={handleChange}
              required
            />

            <input
              type="number"
              name="No_of_Votes"
              placeholder="Number of Votes"
              min="1"
              value={formData.No_of_Votes}
              onChange={handleChange}
              required
            />

            <input
              type="number"
              name="Gross"
              placeholder="Gross Revenue ($)"
              min="0"
              step="any"
              value={formData.Gross}
              onChange={handleChange}
              required
            />

            <input
              type="number"
              name="Runtime"
              placeholder="Runtime (minutes)"
              min="1"
              value={formData.Runtime}
              onChange={handleChange}
              required
            />

            <input
              type="number"
              name="Released_Year"
              placeholder="Release Year"
              value={formData.Released_Year}
              onChange={handleChange}
              required
            />

            <input
              type="text"
              name="Genre"
              placeholder="Genre"
              value={formData.Genre}
              onChange={handleChange}
              required
            />

            <input
              type="text"
              name="Certificate"
              placeholder="Certificate"
              value={formData.Certificate}
              onChange={handleChange}
              required
            />

            <input
              type="text"
              name="Director"
              placeholder="Director"
              value={formData.Director}
              onChange={handleChange}
              required
            />

          </div>

          <textarea
            name="Overview"
            placeholder="Enter movie overview..."
            value={formData.Overview}
            onChange={handleChange}
            required
          />

          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Predict IMDb Rating"}
          </button>

        </form>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {result && (
          <section className="results">

            <h2>Prediction Results</h2>

            <div className="rating">
              {result.predicted_rating}
              <span>/ 10</span>
            </div>

            <h3>
              Predicted Category: {result.predicted_category}
            </h3>

            <p>
              High: {result.confidence.High}%
            </p>

            <p>
              Low: {result.confidence.Low}%
            </p>

            <p>
              Classification Threshold:{" "}
              {result.classification_threshold}
            </p>

          </section>
        )}
        <section className="history-section">

  <h2>Prediction History</h2>

  <div className="history-stats">

    <div className="stat-card">
      <h3>{history.length}</h3>
      <p>Total Predictions</p>
    </div>

    <div className="stat-card">
      <h3>
        {history.filter(movie => movie.actualRating !== null).length}
      </h3>
      <p>Evaluated Movies</p>
    </div>

    <div className="stat-card">
      <h3>
        {history.some(movie => movie.error !== null)
          ? (
              history
                .filter(movie => movie.error !== null)
                .reduce((sum, movie) => sum + Number(movie.error), 0) /
              history.filter(movie => movie.error !== null).length
            ).toFixed(2)
          : "N/A"}
      </h3>
      <p>Average Error (MAE)</p>
    </div>

  </div>

  {history.length === 0 ? (

    <p className="empty-history">
      No predictions yet. Start analyzing movies!
    </p>

  ) : (

    <div className="table-container">

      <table>

        <thead>
          <tr>
            <th>Movie</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Error</th>
            <th>Category</th>
            <th>Action</th>
          </tr>
        </thead>

        <tbody>

          {history.map(movie => (

            <tr key={movie.id}>

              <td>{movie.movieName}</td>

              <td>{movie.predictedRating}/10</td>

              <td>
                {movie.actualRating !== null
                  ? movie.actualRating
                  : (
                    <input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      placeholder="Rating"
                      value={actualRating}
                      onChange={e => setActualRating(e.target.value)}
                    />
                  )}
              </td>

              <td>
                {movie.error !== null
                  ? movie.error
                  : "Pending"}
              </td>

              <td>{movie.category}</td>

              <td>
                {movie.actualRating === null && (
                  <button
                    onClick={() => updateActualRating(movie.id)}
                  >
                    Evaluate
                  </button>
                )}

                <button
                  onClick={() => deletePrediction(movie.id)}
                >
                  Delete
                </button>
              </td>

            </tr>

          ))}

        </tbody>

      </table>

    </div>

  )}

</section>

      </main>

    </div>
  );
}

export default App;