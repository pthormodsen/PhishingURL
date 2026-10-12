import { useState } from "react";
import type { FormEvent } from "react";
import "./App.css";

type Prediction = {
  prediction: 0 | 1;
  label: string;
  model_input: string;
};

const UNAVAILABLE = "The service is not responding right now. Please try again later.";

export default function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      // A proxy error page is not JSON, so check before parsing
      const isJson = response.headers
        .get("Content-Type")
        ?.includes("application/json");
      const body = isJson ? await response.json() : null;

      if (!response.ok) {
        if (response.status === 400 || response.status === 422) {
          throw new Error(
            typeof body?.detail === "string"
              ? body.detail
              : "Enter a valid HTTP or HTTPS URL without spaces."
          );
        }
        throw new Error(UNAVAILABLE);
      }

      if (!body) {
        throw new Error(UNAVAILABLE);
      }

      setResult(body);
    } catch (error) {
      setError(
        // fetch throws a TypeError when the server cannot be reached
        error instanceof Error && !(error instanceof TypeError)
          ? error.message
          : UNAVAILABLE
      );
    } finally {
      setLoading(false);
    }
  }

  const flagged = result?.prediction === 0;

  return (
    <main>
      <h1>Phishing URL Detector</h1>
      <p className="lead">
        Enter a link to see whether a machine learning model finds its URL
        suspicious.
      </p>

      <form onSubmit={handleSubmit}>
        <label htmlFor="url">Website URL</label>
        <div className="row">
          <input
            id="url"
            type="text"
            inputMode="url"
            autoComplete="off"
            autoCapitalize="none"
            spellCheck={false}
            placeholder="example.com/login"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            maxLength={4096}
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      </form>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <section
          className={`result ${flagged ? "flagged" : "neutral"}`}
          aria-live="polite"
        >
          <h2>{result.label}</h2>
          <p>
            {flagged
              ? "The URL resembles phishing links the model was trained on. Be careful before opening it or entering any information."
              : "The model did not find this URL suspicious. This is not a guarantee that the website is safe."}
          </p>
          <p className="model-input">
            The model analyzed: <code>{result.model_input}</code>
          </p>
        </section>
      )}

      <section className="about">
        <h2>How it works</h2>
        <ul>
          <li>
            The model only reads the text of the URL. It never visits the
            website or looks at its content.
          </li>
          <li>
            It was trained on labeled phishing and legitimate URLs from the
            PhreshPhish dataset.
          </li>
          <li>
            It makes mistakes: well-known websites are sometimes flagged, and
            phishing links or fake look-alike domains can pass unflagged.
          </li>
          <li>
            Use the result as an extra warning signal, not as proof that a
            link is safe or dangerous.
          </li>
        </ul>
      </section>
    </main>
  );
}
