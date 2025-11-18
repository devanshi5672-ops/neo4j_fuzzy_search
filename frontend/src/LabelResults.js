import React from "react";

export default function LabelResults({
  labelName,
  results,
  metrics,
  onNext,
  onPrev,
  loading,
  hasPrev,
  hasNext
}) {
  return (
    <div
      style={{
        border: "1px solid #e6e6e6",
        borderRadius: 8,
        padding: 12,
        margin: 8,
        minWidth: 300,
        flex: "1 1 300px",
        background: "#fff",
      }}
    >
      <h3 style={{ marginTop: 0 }}>{labelName}</h3>

      {metrics && (
        <div style={{ fontSize: 12, color: "#444", marginBottom: 8 }}>
          <div>Candidates: {metrics.total_candidates}</div>
          <div>Query time: {metrics.query_time_ms} ms</div>
          <div>
            Neo4j: {metrics.neo4j_time_ms} ms | Rerank: {metrics.rerank_time_ms} ms | Fetch:{" "}
            {metrics.node_fetch_time_ms ?? "-"} ms
          </div>
          <div>Memory: {metrics.memory_used_mb} MB</div>
        </div>
      )}

      {loading && <div>Loading...</div>}

      {!loading && results && results.length > 0 ? (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>
              <th style={{ padding: "6px 4px" }}>Name</th>
              <th style={{ padding: "6px 4px" }}>Labels</th>
              <th style={{ padding: "6px 4px" }}>Score</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.elem_id || r.id} style={{ borderBottom: "1px solid #fafafa" }}>
                <td style={{ padding: "6px 4px" }}>{r.name || r.id}</td>
                <td style={{ padding: "6px 4px" }}>{(r.labels || []).join(", ")}</td>
                <td style={{ padding: "6px 4px" }}>{(r.combined ?? r.score ?? 0).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        !loading && <div style={{ color: "#777" }}>No results</div>
      )}

      {/* Buttons Row */}
      <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between" }}>
        <button
          onClick={onPrev}
          disabled={!hasPrev || loading}
          style={{
            padding: "6px 12px",
            borderRadius: 4,
            border: "none",
            background: hasPrev ? "#ddd" : "#f5f5f5",
            cursor: hasPrev && !loading ? "pointer" : "not-allowed",
          }}
        >
          ⬅ Prev
        </button>

        <button
          onClick={onNext}
          disabled={loading || !hasNext}
        >
          Next ➡
        </button>
      </div>
    </div>
  );
}
