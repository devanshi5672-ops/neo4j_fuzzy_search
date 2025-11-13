import React, { useState } from "react";
import SearchResults from "./SearchResults";
import { searchApi } from "./api";

export default function App() {
  const [query, setQuery] = useState("");
  const [pageSize, setPageSize] = useState(20);
  const [results, setResults] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [cursor, setCursor] = useState(null);
  const [loading, setLoading] = useState(false);

  async function startSearch() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const url = `/search?query=${encodeURIComponent(query)}&page_size=${pageSize}`;
      const res = await searchApi(url);
      setResults(res.results || []);
      setCursor(res.next_cursor || null);
      setMetrics(res.metrics || null);
    } catch (err) {
      console.error(err);
      alert("Search failed. Check console.");
    } finally {
      setLoading(false);
    }
  }

  async function loadNextPage() {
    if (!cursor) return;
    setLoading(true);
    try {
      const params =
        `query=${encodeURIComponent(query)}&page_size=${pageSize}` +
        `&cursor_combined=${cursor.combined}&cursor_id=${encodeURIComponent(cursor.id)}`;
      const res = await searchApi(`/search?${params}`);

      setResults(res.results || []);
      setCursor(res.next_cursor || null);
      setMetrics(res.metrics || null);
    } catch (err) {
      console.error(err);
      alert("Failed to load next page.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Neo4j Fuzzy Search</h1>

      <div className="controls">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search name..."
        />
        <button onClick={startSearch} disabled={loading}>
          Search
        </button>

        <label>
          Page Size:
          <select
            value={pageSize}
            onChange={(e) => setPageSize(Number(e.target.value))}
          >
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
        </label>
      </div>

      <SearchResults results={results} metrics={metrics} />

      <div className="pager">
        <button onClick={loadNextPage} disabled={!cursor || loading}>
          Next Page
        </button>
      </div>
    </div>
  );
}
