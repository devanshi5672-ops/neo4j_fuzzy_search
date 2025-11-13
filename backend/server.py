import os
import time
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase
from rapidfuzz import fuzz

# ---------------- CONFIG ----------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
PORT = int(os.getenv("PORT", "5000"))

INDEXES = ["PlayerNameFT", "TeamNameFT", "TournamentNameFT"]
PAGE_SIZE_DEFAULT = 20
ALPHA = 0.7
WINDOW_LIMIT = 2000
ANALYZER = "standard-folding"  # optional; set to None if unsupported

# ---------------- INIT ----------------
app = Flask(__name__)
CORS(app)
driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
    max_connection_lifetime=3600
)


# ---------------- HELPERS ----------------
def build_lucene_query(q: str) -> str:
    q = q.replace('"', '\\"').strip()
    if not q:
        return ""
    parts = [f"{t}~1" for t in q.split()]
    return "name:(" + " AND ".join(parts) + ")"


def fetch_candidates(session, lucene_query):
    """
    Fetch candidates from each fulltext index.
    Now includes labels(node) so we can dedupe by label set.
    """
    candidates = []
    cypher = """
    CALL db.index.fulltext.queryNodes($index, $lucene, $options)
    YIELD node, score
    RETURN elementId(node) AS elem_id,
           coalesce(node.id, elementId(node)) AS public_id,
           node.name AS name,
           labels(node) AS labels,
           score AS lucene_score
    """
    for idx in INDEXES:
        options = {"limit": WINDOW_LIMIT}
        if ANALYZER:
            options["analyzer"] = ANALYZER
        try:
            res = session.run(cypher, index=idx, lucene=lucene_query, options=options)
            for r in res:
                candidates.append({
                    "elem_id": str(r["elem_id"]),
                    "id": str(r["public_id"]),
                    "name": r["name"],
                    "labels": list(r["labels"]) if r["labels"] is not None else [],
                    "lucene": float(r["lucene_score"] or 0.0)
                })
        except Exception as e:
            print(f"[fetch_candidates] warning: index {idx} -> {e}")
            continue
    return candidates

def rerank_and_sort(candidates, query):
    """
    Re-rank candidates and preserve labels/elem_id/id.
    """
    if not candidates:
        return []
    max_lucene = max((c["lucene"] for c in candidates), default=1) or 1
    reranked = []
    for c in candidates:
        name = c.get("name") or ""
        rf = fuzz.WRatio(query, name)  # 0..100
        combined = ALPHA * (rf / 100.0) + (1 - ALPHA) * (c["lucene"] / max_lucene)
        reranked.append({
            "elem_id": c.get("elem_id"),
            "id": c.get("id"),
            "name": name,
            "labels": c.get("labels", []),
            "combined": float(combined)
        })
    reranked.sort(key=lambda x: (x["combined"], x.get("elem_id", "")), reverse=True)
    return reranked


def fetch_node_details(session, elem_ids):
    """
    Fetch node details using elementId(n) matching.
    'elem_ids' is a list of elementId strings in the desired order.
    Returns a list in same order; missing nodes become placeholders.
    """
    if not elem_ids:
        return []

    cypher = """
    UNWIND $ids AS rid
    MATCH (n)
    WHERE elementId(n) = rid
    RETURN rid AS requested_elem_id, n.name AS name, labels(n) AS labels, n { .id } AS props
    """
    records = session.run(cypher, ids=elem_ids)
    mapping = {}
    for r in records:
        key = str(r["requested_elem_id"])
        mapping[key] = {
            "elem_id": key,
            # prefer stored property id if present in props, else fall back to elem_id
            "id": str(r["props"].get("id")) if r["props"] and "id" in r["props"] else key,
            "name": r["name"],
            "labels": r["labels"] or [],
            "props": r["props"] or {}
        }

    out = []
    missing = []
    for eid in elem_ids:
        s = str(eid)
        if s in mapping:
            out.append(mapping[s])
        else:
            missing.append(s)
            out.append({
                "elem_id": s,
                "id": s,
                "name": None,
                "labels": [],
                "props": {},
                "_missing": True
            })

    if missing:
        print(f"[fetch_node_details] Warning: {len(missing)} requested elementIds not found; sample={missing[:5]}")

    return out

def dedupe_ranked_candidates(ranked, score_precision=4):
    """
    Remove duplicates where (normalized_name, labels_set, rounded_score) are identical.
    Keeps the first occurrence (highest-ranked).
    score_precision: number of decimals to round the combined score to for equality.
    """
    seen = set()
    out = []
    for r in ranked:
        name_norm = (r.get("name") or "").strip().lower()
        labels_key = tuple(sorted([str(x) for x in (r.get("labels") or [])]))
        score_key = round(float(r.get("combined", 0.0)), score_precision)
        key = (name_norm, labels_key, score_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

# ---------------- API ----------------
@app.route("/search", methods=["GET"])
def search():

    # ----- Start performance timer -----
    start_time = time.time()

    q = request.args.get("query", "").strip()
    if not q:
        return jsonify({"error": "query param required"}), 400

    page_size = int(request.args.get("page_size", PAGE_SIZE_DEFAULT))
    # cursor_combined and cursor_id expected to be the combined score and elem_id (elementId string)
    cursor_combined = request.args.get("cursor_combined", type=float)
    cursor_elem_id = request.args.get("cursor_id", type=str)

    lucene_query = build_lucene_query(q)

    # ----- Neo4j fetch candidates -----
    with driver.session() as session:
        neo_start = time.time()
        candidates = fetch_candidates(session, lucene_query)
        neo_time_ms = round((time.time() - neo_start) * 1000, 2)

        # ----- Reranking -----
        rerank_start = time.time()
        ranked = rerank_and_sort(candidates, q)
        rerank_time_ms = round((time.time() - rerank_start) * 1000, 2)
        # dedupe identical (name, labels, rounded_score)
        ranked = dedupe_ranked_candidates(ranked, score_precision=4)

    # ----- Cursor filtering (stateless) -----
    if cursor_combined is not None and cursor_elem_id:
        filtered = []
        for r in ranked:
            # strict lexicographic tie-breaker on elem_id
            if (r["combined"] < cursor_combined) or (r["combined"] == cursor_combined and r.get("elem_id", "") < cursor_elem_id):
                filtered.append(r)
        ranked = filtered

    # ----- Page extraction -----
    page_items = ranked[:page_size]

    next_cursor = None
    if len(ranked) > page_size:
        last = page_items[-1]
        next_cursor = {"combined": last["combined"], "id": last.get("elem_id", last.get("id"))}

    # ----- Fetch node details for the page (by elementId) -----
    elem_ids = [r.get("elem_id") or r.get("id") for r in page_items]

    node_fetch_start = time.time()
    with driver.session() as session:
        details = fetch_node_details(session, elem_ids)
    node_fetch_ms = round((time.time() - node_fetch_start) * 1000, 2)

    # ----- Attach combined score and normalize details -----
    combined_map = {}
    for r in page_items:
        key = r.get("elem_id") or r.get("id")
        combined_map[str(key)] = float(r.get("combined", 0.0))

    normalized_details = []
    for d, requested_eid in zip(details, elem_ids):
        key = str(requested_eid)
        # Ensure stable fields exist
        if "elem_id" not in d:
            d["elem_id"] = key
        if "id" not in d or d["id"] is None:
            d["id"] = d.get("elem_id", key)
        # Always attach numeric combined
        d["combined"] = float(combined_map.get(key, 0.0))
        normalized_details.append(d)

    # ----- Memory usage & timings -----
    process = psutil.Process()
    memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    total_time_ms = round((time.time() - start_time) * 1000, 2)

    metrics = {
        "query_time_ms": total_time_ms,
        "neo4j_time_ms": neo_time_ms,
        "rerank_time_ms": rerank_time_ms,
        "node_fetch_time_ms": node_fetch_ms,
        "total_candidates": len(candidates),
        "returned_results": len(normalized_details),
        "memory_used_mb": memory_mb
    }

    return jsonify({
        "results": normalized_details,
        "next_cursor": next_cursor,
        "metrics": metrics
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
