import re
from typing import List, Tuple, Optional, Any, Dict


def parse_number_token(token: str) -> Optional[float]:
    """Parse a numeric token into float. Handles commas, parentheses (negatives), percent, and dollar signs."""
    if not token or not isinstance(token, str):
        return None
    s = token.strip()
    # Remove currency symbols
    s = s.replace("$", "")
    # Handle parentheses as negative
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    # Percent
    is_percent = False
    if s.endswith("%"):
        is_percent = True
        s = s[:-1]
    # Remove commas and spaces
    s = s.replace(",", "").replace(" ", "")
    try:
        v = float(s)
        if is_percent:
            v = v / 100.0
        if negative:
            v = -v
        return v
    except Exception:
        return None


_NUMBER_RE = re.compile(r"\(?\$?[0-9\,\.]+%?\)?")


def extract_numbers_from_text(text: str) -> List[Tuple[str, float]]:
    """Extract numeric tokens from text and parse them.

    Returns list of (token, value) for tokens that successfully parse.
    """
    if not text:
        return []
    tokens = _NUMBER_RE.findall(text)
    results: List[Tuple[str, float]] = []
    for t in tokens:
        v = parse_number_token(t)
        if v is not None:
            results.append((t, v))
    return results


def find_best_match(value: float, candidates: List[Tuple[str, float]]) -> Optional[Dict[str, Any]]:
    """Find the candidate number closest to value. Returns dict with token, value, diff, rel_error."""
    if not candidates:
        return None
    best = None
    for token, v in candidates:
        diff = abs(v - value)
        rel = diff / (abs(v) + 1e-9)
        if best is None or diff < best["diff"]:
            best = {"token": token, "value": v, "diff": diff, "rel_error": rel}
    return best


def compare_claims_to_context(claims_text: str, context_texts: List[str]) -> List[Dict[str, Any]]:
    """Extract numeric mentions from claims_text and try to verify against numbers found in context_texts.

    Returns list of verification dicts: {claim_token, claim_value, matched, best_source_index, source_token, source_value, diff, rel_error}
    """
    claim_nums = extract_numbers_from_text(claims_text)
    # Build context candidates
    candidates: List[Tuple[str, float, int]] = []  # token, value, source_idx
    for i, ctx in enumerate(context_texts):
        for token, v in extract_numbers_from_text(ctx):
            candidates.append((token, v, i))

    results: List[Dict[str, Any]] = []
    for token, val in claim_nums:
        # Build candidate list for matching
        cand_list = [(t, v) for (t, v, idx) in candidates]
        best = find_best_match(val, cand_list)
        if best:
            # find source idx
            source_idx = next((idx for (t, v, idx) in candidates if t == best["token"] and v == best["value"]), None)
            results.append({
                "claim_token": token,
                "claim_value": val,
                "matched": best["diff"] < max(1e-6, abs(best["value"]) * 0.01),  # within 1% or absolute tiny
                "best_source_index": source_idx,
                "source_token": best["token"],
                "source_value": best["value"],
                "diff": best["diff"],
                "rel_error": best["rel_error"],
            })
        else:
            results.append({
                "claim_token": token,
                "claim_value": val,
                "matched": False,
                "best_source_index": None,
                "source_token": None,
                "source_value": None,
                "diff": None,
                "rel_error": None,
            })
    return results
