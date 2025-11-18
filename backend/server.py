# server.py (UPDATED)
import os
import time
import psutil
import re
from typing import List, Dict, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase

# fuzzy library already used
from rapidfuzz import fuzz

# phonetic library
try:
    import jellyfish
except Exception:
    jellyfish = None

# optional semantic model (lazy)
_USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "0") == "1"
if _USE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.preprocessing import normalize as _normalize
    except Exception:
        # If imports fail, disable embeddings gracefully
        _USE_EMBEDDINGS = False

# ---------------- CONFIG ----------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://13.233.114.24:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Mastut300")
PORT = int(os.getenv("PORT", "5000"))

INDEXES = ["PlayerNameFT", "TeamNameFT", "TournamentNameFT"]
PAGE_SIZE_DEFAULT = 20
ALPHA = float(os.getenv("ALPHA", "0.7"))   # weight between fuzzy and lucene; tuned lower-level later
WINDOW_LIMIT = int(os.getenv("WINDOW_LIMIT", "2000"))  # how many candidates to fetch per index
ANALYZER = os.getenv("ANALYZER", "standard-folding")  # or None

_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# weights for different signals - you can tune via env or code here
DEFAULT_WEIGHTS = {
    "exact": float(os.getenv("W_EXACT", "5.0")),
    "prefix": float(os.getenv("W_PREFIX", "3.0")),
    "fuzzy": float(os.getenv("W_FUZZY", "1.8")),
    "partial": float(os.getenv("W_PARTIAL", "1.0")),
    "phonetic": float(os.getenv("W_PHONETIC", "3.0")),
    "vowel_normal": float(os.getenv("W_VOWEL", "1.5")),
    "semantic": float(os.getenv("W_SEMANTIC", "2.0"))
}

# ---------------- INIT ----------------
app = Flask(__name__)
CORS(app)
driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
    max_connection_lifetime=3600
)

# ---------- Embedding model (lazy, thread-safe) ----------
_embedding_model = None
if _USE_EMBEDDINGS:
    import threading
    _model_lock = threading.Lock()

    def _ensure_embedding_model():
        global _embedding_model
        if _embedding_model is None:
            with _model_lock:
                if _embedding_model is None:
                    _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        return _embedding_model

    def embed_texts(texts: List[str], batch_size: int = 64):
        model = _ensure_embedding_model()
        embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        # L2 normalize for cosine via dot product
        return _normalize(embs, axis=1)
else:
    def embed_texts(texts: List[str], batch_size: int = 64):
        raise RuntimeError("Embeddings disabled - set USE_EMBEDDINGS=1 and install sentence-transformers.")


# ---------------- HELPERS ----------------
def build_lucene_query(q: str) -> str:
    q = q.replace('"', '\\"').strip()
    if not q:
        return ""
    parts = [f"{t}~1" for t in q.split()]
    return "name:(" + " AND ".join(parts) + ")"


def fetch_candidates(session, lucene_query, labels_to_query=None):
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
    idx_list = labels_to_query if labels_to_query else INDEXES
    for idx in idx_list:
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



# ---------- Normalization & phonetic helpers ----------
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def vowel_normalize(s: str) -> str:
    s = normalize_text(s)
    # Replace ee/ii/ie/ei sequences with canonical token IY to allow i -> ee/ii matching
    s = re.sub(r"(ee|ii|ie|ei)+", "IY", s)
    # collapse repeated vowels like aa/oo -> single
    s = re.sub(r"(aa|oo|uu|yy)+", lambda m: m.group(1)[0], s)
    return s


def double_metaphone_codes(s: str) -> Tuple[Optional[str], Optional[str]]:
    s = normalize_text(s)
    if not jellyfish:
        return None, None
    try:
        # jellyfish.metaphone provides a phonetic code (single). If you have a double metaphone
        # library you can use it. We stick with jellyfish.metaphone for portability.
        primary = jellyfish.metaphone(s)
        # secondary not available via jellyfish; return None for secondary
        return primary, None
    except Exception:
        return None, None


# ---------- Reranker (combines fuzzy, phonetic, vowel, optional semantic) ----------
def rerank_candidates(candidates: List[Dict], query: str,
                      key_name: str = "name",
                      use_embeddings: bool = _USE_EMBEDDINGS,
                      precomputed_emb_key: Optional[str] = "embedding",
                      weights: Optional[Dict] = None,
                      max_candidates: Optional[int] = None) -> List[Dict]:
    """
    candidates: list of dicts; each must have at least the name property (key_name).
    returns candidates sorted by combined_score and attaches _signals for debugging.
    """
    if not candidates:
        return []

    if weights is None:
        weights = DEFAULT_WEIGHTS

    q_norm = normalize_text(query)
    q_vn = vowel_normalize(query)
    q_codes = double_metaphone_codes(query)

    # prepare candidate arrays
    names = []
    names_vn = []
    cand_codes = []
    emb_candidates = None
    use_precomputed_emb = False

    for c in candidates:
        n_raw = c.get(key_name) or ""
        # prefer precomputed lower if available
        n = c.get("name_lower") or normalize_text(n_raw)
        names.append(n)
        names_vn.append(c.get("name_vowel") or vowel_normalize(n))
        # try to use precomputed metaphone if present
        if c.get("metaphone"):
            cand_codes.append((c.get("metaphone"), None))
        else:
            cand_codes.append(double_metaphone_codes(n))

    # embeddings (optional)
    if use_embeddings:
        if precomputed_emb_key and all(precomputed_emb_key in c and c[precomputed_emb_key] for c in candidates):
            try:
                import numpy as _np
                emb_candidates = _np.vstack([_np.array(c[precomputed_emb_key], dtype=_np.float32) for c in candidates])
                emb_candidates = _normalize(emb_candidates, axis=1)
                use_precomputed_emb = True
            except Exception:
                emb_candidates = None
                use_precomputed_emb = False

        if not use_precomputed_emb:
            try:
                emb_candidates = embed_texts(names)
            except Exception as e:
                print("[rerank_candidates] embedding failed:", e)
                emb_candidates = None

    emb_query = None
    if use_embeddings:
        try:
            emb_query = embed_texts([q_norm])[0]
        except Exception:
            emb_query = None

    # compute fuzzy + phonetic + vowel signals
    scored = []
    for idx, c in enumerate(candidates):
        name = names[idx]
        name_vn = names_vn[idx]
        codes = cand_codes[idx]

        signals = {}
        # exact match
        signals["exact"] = 1.0 if (q_norm == name) else 0.0
        # prefix or token overlap
        signals["prefix"] = 1.0 if name.startswith(q_norm) or q_norm.startswith(name) else 0.0
        # fuzzy & partial fuzzy (0..1)
        try:
            signals["fuzzy"] = fuzz.WRatio(q_norm, name) / 100.0
        except Exception:
            signals["fuzzy"] = 0.0
        try:
            signals["partial"] = fuzz.partial_ratio(q_norm, name) / 100.0
        except Exception:
            signals["partial"] = 0.0

        # vowel-normalized signals
        signals["vowel_exact"] = 1.0 if q_vn == name_vn else 0.0
        try:
            signals["vowel_fuzzy"] = fuzz.WRatio(q_vn, name_vn) / 100.0
        except Exception:
            signals["vowel_fuzzy"] = 0.0

        # phonetic scoring
        if q_codes and codes and q_codes[0] and codes[0]:
            # primary match strong
            if q_codes[0] == codes[0]:
                signals["phonetic"] = 1.0
            elif q_codes[0] == codes[1] or (q_codes[1] and q_codes[1] == codes[0]):
                signals["phonetic"] = 0.9
            else:
                signals["phonetic"] = 0.0
        else:
            signals["phonetic"] = 0.0

        # semantic cosine
        if emb_query is not None and emb_candidates is not None:
            try:
                import numpy as _np
                sim = float(_np.dot(emb_candidates[idx], emb_query))
                signals["semantic"] = max(0.0, sim)
            except Exception:
                signals["semantic"] = 0.0
        else:
            signals["semantic"] = 0.0

        # Now combine weighted sum
        combined = 0.0
        combined += weights.get("exact", 0.0) * signals["exact"]
        combined += weights.get("prefix", 0.0) * signals["prefix"]
        combined += weights.get("fuzzy", 0.0) * signals["fuzzy"]
        combined += weights.get("partial", 0.0) * signals["partial"]
        # vowel_normal weight applies to exact and fuzzy variant
        combined += weights.get("vowel_normal", 0.0) * (signals["vowel_exact"] * 1.0 + signals["vowel_fuzzy"] * 0.6)
        combined += weights.get("phonetic", 0.0) * signals["phonetic"]
        combined += weights.get("semantic", 0.0) * signals["semantic"]

        # include lucene score as secondary signal normalized later by caller (we will incorporate later)
        c_out = dict(c)  # shallow copy
        c_out["_signals"] = signals
        c_out["combined"] = float(combined)
        scored.append(c_out)

    # incorporate lucene score (scale and add) - normalize lucene to 0..1 using max
    max_lucene = max((c.get("lucene", 0.0) for c in candidates), default=1.0) or 1.0
    for s in scored:
        luc = float(s.get("lucene", 0.0))
        # combine with ALPHA (keeps backwards compatibility): final = ALPHA * semantic_fuzzy + (1-ALPHA) * (lucene/max)
        # Here we interpret combined as "semantic+signals" and blend with lucene
        s["combined"] = float(ALPHA * s["combined"] + (1 - ALPHA) * (luc / max_lucene))

    # sort by combined desc; tie-break by elem_id
    scored.sort(key=lambda x: (x.get("combined", 0.0), x.get("elem_id", "")), reverse=True)

    if max_candidates:
        return scored[:max_candidates]
    return scored


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

    # handle labels param: comma-separated list of INDEX names or label names
    labels_param = request.args.get("labels", "")  # example: "Player" or "Player,Team"
    labels_list = None
    if labels_param:
        requested = [s.strip() for s in labels_param.split(",") if s.strip()]
        label_to_index = {
            "Player": "PlayerNameFT",
            "Team": "TeamNameFT",
            "Tournament": "TournamentNameFT"
        }
        labels_list = []
        for r in requested:
            if r in label_to_index.values():
                labels_list.append(r)
            elif r in label_to_index:
                labels_list.append(label_to_index[r])
            else:
                labels_list.append(r)

    # ----- Neo4j fetch candidates -----
    with driver.session() as session:
        neo_start = time.time()
        candidates = fetch_candidates(session, lucene_query, labels_to_query=labels_list)
        neo_time_ms = round((time.time() - neo_start) * 1000, 2)

        # ----- Reranking -----
        rerank_start = time.time()
        # fetch a broader candidate pool already controlled by WINDOW_LIMIT; rerank and trim to page_size
        ranked = rerank_candidates(candidates, q, key_name="name", use_embeddings=_USE_EMBEDDINGS, precomputed_emb_key="embedding", weights=None, max_candidates=None)
        rerank_time_ms = round((time.time() - rerank_start) * 1000, 2)

        # dedupe identical (name, labels, rounded_score)
        ranked = dedupe_ranked_candidates(ranked, score_precision=4)

    # ----- Cursor filtering (stateless) -----
    if cursor_combined is not None and cursor_elem_id:
        filtered = []
        for r in ranked:
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
        if "elem_id" not in d:
            d["elem_id"] = key
        if "id" not in d or d["id"] is None:
            d["id"] = d.get("elem_id", key)
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
