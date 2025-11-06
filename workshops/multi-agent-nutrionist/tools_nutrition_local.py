# tools_nutrition_local.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import re, json

TSV_PATH = Path("opennutrition_foods.tsv")
JSON_COLS = ["alternate_names","source","serving","nutrition_100g",
             "labels","ingredients","ingredient_analysis"]

def _safe_json(x):
    if x is None or str(x).strip() == "":
        return None
    s = str(x).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return s

def _load_tsv(tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    for c in JSON_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_safe_json)

    def make_search(row):
        names = [row.get("name","")]
        alts = row.get("alternate_names")
        if isinstance(alts, list):
            names += alts
        elif isinstance(alts, str):
            names.append(alts)
        return " | ".join([n for n in names if n]).lower()

    df["_search_text"] = df.apply(make_search, axis=1)
    df["_ean"] = df.get("ean_13", "").astype(str).str.replace(r"\D", "", regex=True)
    return df

# Lazy-loaded global to avoid import-time failures
_DF: Optional[pd.DataFrame] = None

def _ensure_df():
    """Ensure the module-global DataFrame is loaded."""
    global _DF
    if _DF is None:
        if not TSV_PATH.exists():
            raise FileNotFoundError(f"TSV file not found at {TSV_PATH.resolve()}")
        _DF = _load_tsv(TSV_PATH)

def _norm_ingredients(ing) -> Optional[List[str]]:
    if ing is None:
        return None
    if isinstance(ing, list):
        return [re.sub(r"\s+", " ", str(i)).strip(" ,;") for i in ing if str(i).strip()] or None
    if isinstance(ing, str):
        parts = re.split(r"[;,]\s*", ing.strip())
        return [re.sub(r"\s+", " ", p).strip(" ,;") for p in parts if p] or None
    return None

def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    n = row.get("nutrition_100g")
    if isinstance(n, dict):
        keep = ["calories","protein","total_fat","saturated_fats",
                "carbohydrates","total_sugars","sodium","dietary_fiber"]
        n = {k: n.get(k) for k in keep if k in n}
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "ean_13": row.get("ean_13"),
        "ingredients_raw": row.get("ingredients"),
        "ingredients_list": _norm_ingredients(row.get("ingredients")),
        "serving": row.get("serving"),
        "nutrition_100g": n,
        "source": row.get("source"),
    }

def lookup_by_barcode_local(ean_13: str) -> Optional[Dict[str, Any]]:
    _ensure_df()
    code = re.sub(r"\D", "", str(ean_13))
    hits = _DF[_DF["_ean"] == code]  # type: ignore[index]
    return None if hits.empty else _payload(hits.iloc[0].to_dict())

def lookup_by_name_local(name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    _ensure_df()
    q = name.lower().strip()
    def score(txt: str) -> float:
        if not txt:
            return 0.0
        if q in txt:
            return 1.0 + len(q) / max(1, len(txt))
        tq = set(re.findall(r"\w+", q)); tt = set(re.findall(r"\w+", txt))
        return 0.0 if not tq or not tt else len(tq & tt) / len(tq | tt)
    s = _DF["_search_text"].apply(score)  # type: ignore[index]
    hits = _DF.loc[s > 0].copy()          # type: ignore[index]
    hits["_score"] = s[s > 0]
    hits.sort_values("_score", ascending=False, inplace=True)
    return [_payload(r.to_dict()) for _, r in hits.head(top_k).iterrows()]

# --- Public helpers -------------------------------------------------

def load_tsv(tsv: Union[str, Path]) -> pd.DataFrame:
    """Public loader for a TSV file that matches this schema."""
    return _load_tsv(Path(tsv))

def reload_tsv(tsv: Union[str, Path]) -> int:
    """Replace the module-global DF with a fresh one; returns row count."""
    global _DF, TSV_PATH
    TSV_PATH = Path(tsv)
    _DF = _load_tsv(TSV_PATH)
    return len(_DF)

__all__ = [
    "load_tsv",
    "reload_tsv",
    "lookup_by_barcode_local",
    "lookup_by_name_local",
]
