# reranker.py
from typing import List, Dict, Optional, Tuple
import numpy as np
from rapidfuzz import fuzz
import jellyfish
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import threading

# Optional: semantic model (lazy load)
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_model_lock = threading.Lock()

def _ensure_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(_MODEL_NAME)
    return _model

# ---------- Normalization helpers ----------
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Vowel normalization: map repeated e/i combos to a canonical token to handle "ee"/"ii" vs "i"
# We choose to replace "ee", "ii", "ie", "ei" sequences with a canonical "IY" token for matching.
def vowel_normalize(s: str) -> str:
    s = normalize_text(s)
    # common patterns where user might type 'i' instead of 'ee' or 'ii'
    # Replace ee, ii, ie, ei -> special marker
    s = re.sub(r"(ee|ii|ie|ei)+", "IY", s)
    # Also collapse repeated vowels to single
    s = re.sub(r"(aa|oo|uu|yy)+", lambda m: m.group(1)[0], s)
    return s

def double_metaphone_codes(s: str) -> Tuple[Optional[str], Optional[str]]:
    s = normalize_text(s)
    try:
        primary, secondary = jellyfish.metaphone(s), None
        # jellyfish has 'metaphone' (single), for double metaphone import from jellyfish? 
        # jellyfish.metaphone approximates - it's acceptable; if you want true Double Metaphone consider 'fuzzy' library.
        # We'll use metaphone as a compact phonetic code.
        return primary, secondary
    except Exception:
        return None, None

# ---------- Embedding helper ----------
def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = _ensure_model()
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return normalize(embs, axis=1)

# ---------- Scoring functions ----------
def fuzzy_score(a: str, b: str) -> float:
    """0..100 RapidFuzz ratio; convert to 0..1"""
    try:
        return fuzz.WRatio(a, b) / 100.0
    except Exception:
        return 0.0

def partial_fuzzy_score(a: str, b: str) -> float:
    try:
        return fuzz.partial_ratio(a, b) / 100.0
    except Exception:
        return 0.0

def phonetic_score(q_codes: Tuple[Optional[str], Optional[str]], cand_codes: Tuple[Optional[str], Optional[str]]) -> float:
    """Return 1.0 if primary codes match, 0.75 if secondary matches, else 0"""
    if not q_codes or not cand_codes: return 0.0
    q1, q2 = q_codes
    c1, c2 = cand_codes
    if q1 and c1 and q1 == c1:
        return 1.0
    if q1 and c2 and q1 == c2:
        return 0.9
    if q2 and c1 and q2 == c1:
        return 0.85
    return 0.0

# ---------- Combined reranker ----------
def rerank_candidates(candidates: List[Dict],
                      query: str,
                      key_name: str = "name",
                      use_embeddings: bool = True,
                      precomputed_emb_key: Optional[str] = "embedding",
                      weights: Optional[Dict] = None,
                      max_candidates: Optional[int] = None) -> List[Dict]:
    """
    candidates: list of dicts; each must have at least the name property (key_name).
    query: user string
    weights: dictionary of weights for signals. Example:
      {
        "exact": 3.0,
        "prefix": 2.0,
        "fuzzy": 1.5,
        "partial": 1.0,
        "phonetic": 2.5,
        "vowel_normal": 1.2,
        "semantic": 2.0
      }
    returns candidates sorted by combined_score. Each candidate will get fields:
       combined_score, signals: { fuzzy, phonetic, semantic, ... }
    """

    if weights is None:
        weights = {
            "exact": 5.0,
            "prefix": 3.0,
            "fuzzy": 1.8,
            "partial": 1.0,
            "phonetic": 3.0,
            "vowel_normal": 1.5,
            "semantic": 2.0
        }

    q_norm = normalize_text(query)
    q_vn = vowel_normalize(query)
    q_codes = double_metaphone_codes(query)

    # Prepare candidate arrays
    names = []
    names_vn = []
    cand_codes = []
    emb_candidates = None
    use_precomputed_emb = False

    for c in candidates:
        n = normalize_text(c.get(key_name, "") or "")
        names.append(n)
        names_vn.append(vowel_normalize(n))
        cand_codes.append(double_metaphone_codes(n))

    # optional semantic embeddings
    if use_embeddings:
        # If candidates have precomputed embeddings as lists, use them
        if precomputed_emb_key and all(precomputed_emb_key in c and c[precomputed_emb_key] for c in candidates):
            try:
                emb_candidates = np.vstack([np.array(c[precomputed_emb_key], dtype=np.float32) for c in candidates])
                emb_candidates = normalize(emb_candidates, axis=1)
                use_precomputed_emb = True
            except Exception:
                emb_candidates = None
                use_precomputed_emb = False

        if not use_precomputed_emb:
            try:
                emb_candidates = embed_texts(names)
            except Exception:
                emb_candidates = None

    emb_query = None
    if use_embeddings:
        try:
            emb_query = embed_texts([q_norm])[0]
        except Exception:
            emb_query = None

    # compute signals and combined score
    scored = []
    for idx, c in enumerate(candidates):
        name = names[idx]
        name_vn = names_vn[idx]
        codes = cand_codes[idx]

        signals = {}
        # exact match
        signals["exact"] = 1.0 if (q_norm == name) else 0.0
        # prefix (query begins the name) - useful for typed-in names
        signals["prefix"] = 1.0 if name.startswith(q_norm) or q_norm.startswith(name) else 0.0
        # fuzzy and partial fuzzy
        signals["fuzzy"] = fuzzy_score(q_norm, name)
        signals["partial"] = partial_fuzzy_score(q_norm, name)
        # vowel-normalized fuzzy (handle i vs ee/ii)
        signals["vowel_exact"] = 1.0 if q_vn == name_vn else 0.0
        signals["vowel_fuzzy"] = fuzzy_score(q_vn, name_vn)

        # phonetic
        signals["phonetic"] = phonetic_score(q_codes, codes)

        # semantic (cosine)
        if emb_query is not None and emb_candidates is not None:
            try:
                sim = float(np.dot(emb_candidates[idx], emb_query))
                # map from -1..1 to 0..1 (but normalized embeddings give -1..1 rarely)
                signals["semantic"] = max(0.0, sim)
            except Exception:
                signals["semantic"] = 0.0
        else:
            signals["semantic"] = 0.0

        # Compose combined score (weighted sum)
        combined = 0.0
        combined += weights.get("exact", 0.0) * signals["exact"]
        combined += weights.get("prefix", 0.0) * signals["prefix"]
        combined += weights.get("fuzzy", 0.0) * signals["fuzzy"]
        combined += weights.get("partial", 0.0) * signals["partial"]
        combined += weights.get("vowel_normal", 0.0) * (signals["vowel_exact"] * 1.0 + signals["vowel_fuzzy"] * 0.6)
        combined += weights.get("phonetic", 0.0) * signals["phonetic"]
        combined += weights.get("semantic", 0.0) * signals["semantic"]

        c_out = dict(c)  # copy
        c_out["_signals"] = signals
        c_out["combined_score"] = float(combined)
        scored.append(c_out)

    # sort by combined score
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    if max_candidates:
        return scored[:max_candidates]
    return scored
