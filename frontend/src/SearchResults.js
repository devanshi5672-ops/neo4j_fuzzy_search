import React from "react";

export default function SearchResults({ results, metrics }) {

  return (
    <div>
      {/* Metrics Section */}
      {metrics && (
        <div className="metrics">
          <h3>Performance Metrics</h3>
          <p><b>Total Candidates:</b> {metrics.total_candidates}</p>
          <p><b>Returned Results:</b> {metrics.returned_results}</p>
          <p><b>Total Query Time:</b> {metrics.query_time_ms} ms</p>
          <p><b>Neo4j Fetch Time:</b> {metrics.neo4j_time_ms} ms</p>
          <p><b>Rerank Time:</b> {metrics.rerank_time_ms} ms</p>
          <p><b>Backend Memory Usage:</b> {metrics.memory_used_mb} MB</p>
        </div>
      )}

      {/* Results Table */}
      {results && results.length > 0 ? (
        <table className="results-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Labels</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.name}</td>
                <td>{(r.labels || []).join(", ")}</td>
                <td>{typeof r.combined === "number" ? r.combined.toFixed(4) : (Number(r.combined || 0).toFixed(4))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No results yet.</p>
      )}
    </div>
  );
}
