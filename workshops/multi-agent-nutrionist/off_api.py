# off_api.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time, httpx

OFF_BASE = "https://world.openfoodfacts.org"   # use .net for staging

class OFFClient:
    def __init__(
        self,
        base_url: str = OFF_BASE,
        user_agent: str = "IngredientOrchestrator/1.0 (you@example.com)",
        timeout: float = 10.0,
    ):
        self.base = base_url.rstrip("/")
        self.session = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )

    def _get(self, path: str, params: Dict[str, Any] | None = None, max_retries=3):
        url = f"{self.base}{path}"
        for i in range(max_retries + 1):
            r = self.session.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504) and i < max_retries:
                time.sleep(0.5 * (2 ** i))  # simple backoff
                continue
            r.raise_for_status()
            return r

    def product_by_barcode(self, ean13: str) -> Optional[Dict[str, Any]]:
        ean = "".join(ch for ch in str(ean13) if ch.isdigit())
        if not ean:
            return None
        r = self._get(f"/api/v2/product/{ean}.json")
        data = r.json()
        prod = data.get("product") or (data.get("products") or [None])[0]
        return self._normalize(prod) if prod else None

    def search_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "page_size": min(max(limit, 1), 20),  # keep small to respect 10 req/min
            "fields": "code,product_name,brands,ingredients_text,last_modified_t,countries_tags",
        }
        r = self._get("/api/v2/search", params=params)
        prods = r.json().get("products") or []
        return [self._normalize(p) for p in prods if p]

    @staticmethod
    def _normalize(p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "barcode": p.get("code"),
            "product_name": p.get("product_name") or p.get("generic_name"),
            "brand": (p.get("brands") or "").split(",")[0].strip() or None,
            "ingredients_text": p.get("ingredients_text"),
            "countries_tags": p.get("countries_tags"),
            "last_modified_t": p.get("last_modified_t"),
            "source": "off",
        }
