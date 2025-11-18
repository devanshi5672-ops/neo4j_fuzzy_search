import React, { useState, useEffect } from "react";
import LabelResults from "./LabelResults";

const API_BASE = process.env.REACT_APP_API_BASE || "http://192.168.4.11:5000";
const PAGE_SIZE_DEFAULT = 20;
const LABELS = ["Player", "Team", "Tournament"]; // for UI toggles

export default function App() {
  const [query, setQuery] = useState("");
  const [pageSize, setPageSize] = useState(PAGE_SIZE_DEFAULT);

  // selectedLabels is a set of labels the user wants to see
  const [selectedLabels, setSelectedLabels] = useState(new Set(["Player"]));

  // per-label state:
  // pages: array of pages (each page: { results, cursor, metrics })
  // current: index into pages (0-based)
  // loading: boolean
  const emptyLabel = { pages: [], current: -1, loading: false };
  const [labelState, setLabelState] = useState({
    Player: { ...emptyLabel },
    Team: { ...emptyLabel },
    Tournament: { ...emptyLabel },
  });

  useEffect(() => {
    // When selection changes, optionally run searches for newly selected labels if query present
    if (query.trim()) {
      selectedLabels.forEach(lbl => fetchLabel(lbl, { reset: true }));
    }
    // eslint-disable-next-line
  }, [selectedLabels]);

  function toggleLabel(label) {
    const next = new Set(selectedLabels);
    if (next.has(label)) next.delete(label); else next.add(label);
    setSelectedLabels(next);
  }

  // helper to set labelState for a single label
  function updateLabel(label, patch) {
    setLabelState(s => ({ ...s, [label]: { ...s[label], ...patch } }));
  }

  async function fetchLabel(label, options = {}) {
    // options: { reset: boolean } if reset fetches first page (clears history)
    updateLabel(label, { loading: true });
    try {
      const lblState = labelState[label] || { pages: [], current: -1 };
      const currentPage = lblState.current >= 0 ? lblState.pages[lblState.current] : null;
      const cursor = options.reset ? null : (currentPage ? currentPage.cursor : null);

      const params = new URLSearchParams();
      params.append("query", query);
      params.append("page_size", pageSize);
      params.append("labels", label);
      if (cursor && cursor.combined != null && cursor.id) {
        params.append("cursor_combined", cursor.combined);
        params.append("cursor_id", cursor.id);
      }
      const url = `${API_BASE}/search?${params.toString()}`;
      const res = await fetch(url, { headers: { "x-api-key": process.env.REACT_APP_API_KEY || "" } });
      if (!res.ok) {
        console.error("Search error", await res.text());
        updateLabel(label, { loading: false });
        return;
      }
      const j = await res.json();

      // New page object returned from backend
      const newPage = {
        results: j.results || [],
        cursor: j.next_cursor || null,
        metrics: j.metrics || null,
      };

      setLabelState(s => {
        const prev = s[label] || { pages: [], current: -1 };
        let pages;
        let current;
        if (options.reset) {
          // start fresh
          pages = [newPage];
          current = 0;
        } else {
          // push current page to history if current exists, then append new page
          // But if current === -1 (no pages yet), treat as first page
          pages = prev.pages.slice(0, prev.current + 1); // trim any forward pages
          pages.push(newPage);
          current = pages.length - 1;
        }
        return { ...s, [label]: { ...prev, pages, current, loading: false } };
      });
    } catch (e) {
      console.error(e);
      updateLabel(label, { loading: false });
    }
  }

  // called when user clicks search: run search (reset) for every selected label
  async function onSearchClick() {
    for (const lbl of selectedLabels) {
      // reset = true to start from first page
      await fetchLabel(lbl, { reset: true });
    }
  }

  // Next button: fetch next page (calls backend, which returns next_cursor)
  function onNextForLabel(label) {
    // fetchLabel with reset=false will append a new page based on current cursor
    fetchLabel(label, { reset: false });
  }

  // Prev button: restore previous page from pages history (no backend call)
  function onPrevForLabel(label) {
    setLabelState(s => {
      const prev = s[label];
      if (!prev || prev.current <= 0) return s; // nothing to do
      const nextCurrent = prev.current - 1;
      const newLabelState = {
        ...s,
        [label]: { ...prev, current: nextCurrent }
      };
      return newLabelState;
    });
  }

  // Helper to read the current page for rendering
  function getPageForLabel(label) {
    const st = labelState[label];
    if (!st || st.current < 0) return { results: [], metrics: null, loading: false, hasPrev: false, hasNext: false };
    const page = st.pages[st.current];
    const hasPrev = st.current > 0;
    // hasNext: if current page has a cursor (meaning backend can provide next)
    const hasNext = !!(page && page.cursor && (page.metrics ? page.metrics.returned_results > 0 : true));
    return { results: page.results, metrics: page.metrics, loading: st.loading, hasPrev, hasNext };
  }

  return (
    <div style={{ padding: 20 }}>
      <h1>Neo4j Fuzzy Search — Per-label panels</h1>

      <div style={{ marginBottom: 12 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search name..."
          style={{ padding: 8, width: 320 }}
        />
        <button onClick={onSearchClick} disabled={!query.trim()} style={{ marginLeft: 8 }}>Search</button>

        <label style={{ marginLeft: 16 }}>
          Page Size:
          <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))} style={{ marginLeft: 6 }}>
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
        </label>
      </div>

      <div style={{ marginBottom: 16 }}>
        {LABELS.map(lbl => (
          <button
            key={lbl}
            onClick={() => toggleLabel(lbl)}
            style={{
              marginRight: 8,
              padding: "6px 12px",
              background: selectedLabels.has(lbl) ? "#4CAF50" : "#eee",
              color: selectedLabels.has(lbl) ? "#fff" : "#333",
              border: "none",
              borderRadius: 4,
              cursor: "pointer"
            }}
          >
            {lbl}
          </button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        {Array.from(selectedLabels).map(lbl => {
          const page = getPageForLabel(lbl);
          return (
            <LabelResults
              key={lbl}
              labelName={lbl}
              results={page.results}
              metrics={page.metrics}
              loading={page.loading}
              onNext={() => onNextForLabel(lbl)}
              onPrev={() => onPrevForLabel(lbl)}
              hasPrev={page.hasPrev}
              hasNext={page.hasNext}
            />
          );
        })}

        {selectedLabels.size === 0 && <div style={{ color: "#777" }}>Select at least one label to show results</div>}
      </div>
    </div>
  );
}
