import { useState } from "react";

// Change this if your backend runs on a different port/host,
// or set it via an environment variable when you deploy.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const EXAMPLES = [
  "how do I get my money back",
  "I forgot my login credentials",
  "is it safe to pay here",
  "using this on my phone",
];

function ResultCard({ result, mode, maxScore }) {
  const pct = maxScore > 0 ? Math.max(4, (result.score / maxScore) * 100) : 0;
  return (
    <div className="result-card">
      <p className="result-q">{result.question}</p>
      <p className="result-a">{result.answer}</p>
      <div className="score-row">
        <div className="score-bar-track">
          <div
            className={`score-bar-fill ${mode}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="score-value">
          {mode === "semantic" ? result.score.toFixed(3) : result.score}
        </span>
      </div>
    </div>
  );
}

function ResultColumn({ title, mode, results, loading }) {
  const maxScore = results.length ? results[0].score : 0;
  return (
    <div>
      <div className="column-label">
        <span className={`dot ${mode}`} />
        {title}
      </div>
      {loading && <p className="empty-state">Searching…</p>}
      {!loading && results.length === 0 && (
        <p className="empty-state">
          {mode === "keyword"
            ? "No results — none of these FAQs share a matching word."
            : "No results yet. Try a search above."}
        </p>
      )}
      {!loading &&
        results.map((r) => (
          <ResultCard key={r.id} result={r} mode={mode} maxScore={maxScore} />
        ))}
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("compare"); // 'semantic' | 'keyword' | 'compare'
  const [semanticResults, setSemanticResults] = useState([]);
  const [keywordResults, setKeywordResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  async function runSearch(q) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setHasSearched(true);
    try {
      const res = await fetch(
        `${API_URL}/search/compare?q=${encodeURIComponent(q)}&top_k=5`
      );
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setSemanticResults(data.semantic_results);
      setKeywordResults(data.keyword_results);
    } catch (err) {
      setError(
        "Couldn't reach the search API. Is the backend running on " +
          API_URL +
          "?"
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    runSearch(query);
  }

  function handleExampleClick(ex) {
    setQuery(ex);
    runSearch(ex);
  }

  return (
    <div className="page">
      <div className="hero">
        <h1>Search by meaning, not keywords.</h1>
        <p>
          This searches a small FAQ set using sentence embeddings and cosine
          similarity, instead of matching exact words. Try a query that
          doesn't share any words with the answer you're looking for.
        </p>
      </div>

      <form className="search-row" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask something, in your own words…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" onClick={() => handleExampleClick(ex)}>
            {ex}
          </button>
        ))}
      </div>

      {error && <p className="status-line error">{error}</p>}

      {hasSearched && (
        <div className="results-columns">
          <ResultColumn
            title="semantic search"
            mode="semantic"
            results={semanticResults}
            loading={loading}
          />
          <ResultColumn
            title="keyword search"
            mode="keyword"
            results={keywordResults}
            loading={loading}
          />
        </div>
      )}

      <div className="footer-note">
        Semantic search embeds your query and every FAQ into vectors, then
        ranks results by cosine similarity — how close their directions are
        in vector space. Keyword search only counts shared words. That's why
        semantic search can find "reset my password" when you type "I forgot
        my login credentials," even though those sentences share almost no
        words.
      </div>
    </div>
  );
}
