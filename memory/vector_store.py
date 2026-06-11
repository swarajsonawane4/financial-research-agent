"""Long-term memory — a Chroma vector database for research findings.

This is the agent's persistent, cross-session memory. When the agent finishes
researching something, it stores the finding here as an embedded vector with
metadata (ticker, source type, date, confidence). Before starting new research,
it queries this store to retrieve relevant past findings by MEANING, not keyword.

This is what Challenge 7 ("based on companies you've already researched, what
themes emerge?") depends on — the agent recalls everything it has learned across
all prior runs.

Design follows the reference document's vector-DB schema (Section A3.3): each
record carries id, content, ticker, source_type, date, confidence, session, and
a verified flag. Chroma runs locally (no API key, no cost) and persists to disk,
so memory survives between runs. Embeddings use a local sentence-transformers
model (also free / offline after first download).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover
    chromadb = None
    embedding_functions = None


# Where the vector DB persists on disk (survives between runs).
DB_DIR = Path.home() / ".cache" / "financial-research-agent" / "chroma"
COLLECTION_NAME = "research_findings"

# Local, free embedding model. ~80MB, downloaded once on first use.
EMBED_MODEL = "all-MiniLM-L6-v2"


class VectorMemory:
    """Wrapper around a Chroma collection of research findings."""

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Run: "
                "pip install chromadb sentence-transformers"
            )
        persist_dir = persist_dir or DB_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        # get_or_create so we reuse the same collection across runs.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # -- write --
    def store(
        self,
        content: str,
        *,
        ticker: str = "",
        source_type: str = "analysis",
        date: Optional[str] = None,
        confidence: float = 0.8,
        session: str = "",
        verified: bool = False,
    ) -> dict:
        """Store a research finding. Returns {"ok": True, "id": ...}.

        A deterministic id (hash of ticker+content) prevents storing the exact
        same finding twice — re-storing just overwrites the existing record.
        """
        if not content.strip():
            return {"ok": False, "error": "Cannot store empty content."}

        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # stable id from ticker+content so duplicates collapse
        digest = hashlib.sha1(f"{ticker}:{content}".encode()).hexdigest()[:16]
        doc_id = f"{ticker.lower() or 'gen'}-{digest}"

        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{
                "ticker": ticker.upper(),
                "source_type": source_type,
                "date": date,
                "confidence": float(confidence),
                "session": session or str(uuid.uuid4())[:8],
                "verified": bool(verified),
            }],
        )
        return {"ok": True, "id": doc_id}

    # -- read --
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter: Optional[dict] = None,
        min_similarity: float = 0.05,
    ) -> dict:
        """Semantic search over stored findings.

        Args:
            query: natural-language query.
            top_k: how many results to return.
            filter: optional metadata filter, e.g. {"ticker": "TSLA"}.
            min_similarity: drop results below this similarity (0-1). Guards
                against returning weak, noise-level matches. When a metadata
                filter is supplied (e.g. ticker), the threshold is relaxed,
                since the filter already guarantees relevance to that company.

        Returns a dict with a `results` list of
        {content, ticker, source_type, date, confidence, similarity}.
        """
        if self.count() == 0:
            return {"ok": True, "results": [], "note": "Memory is empty."}

        # Chroma's where-filter wants uppercase tickers to match what we store.
        where = None
        if filter:
            where = {k: (v.upper() if k == "ticker" and isinstance(v, str) else v)
                     for k, v in filter.items()}

        res = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self.count()),
            where=where,
        )

        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            similarity = round(1 - dist, 3)
            out.append({
                "content": doc,
                "ticker": meta.get("ticker", ""),
                "source_type": meta.get("source_type", ""),
                "date": meta.get("date", ""),
                "confidence": meta.get("confidence"),
                # cosine distance -> similarity (1 = identical)
                "similarity": similarity,
            })

        # When a metadata filter (e.g. ticker) is present, it has already
        # guaranteed the results are about the right company — so trust it and
        # keep everything; the fuzzy similarity score shouldn't override an
        # explicit ticker match. Only apply the threshold to UNFILTERED searches,
        # where cross-topic noise is the real risk.
        if where:
            filtered = out
        else:
            filtered = [r for r in out if r["similarity"] >= min_similarity]

        return {"ok": True, "results": filtered}

    # -- utilities --
    def count(self) -> int:
        """How many findings are currently stored."""
        return self._collection.count()

    def reset(self) -> None:
        """Wipe all stored findings (useful for clean test runs)."""
        self._client.delete_collection(COLLECTION_NAME)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )


# --- Module-level singleton + tool-shaped wrappers ---------------------------
# The registry tools (vector_db_search / vector_db_store) call these.

_memory: Optional[VectorMemory] = None


def _get_memory() -> VectorMemory:
    global _memory
    if _memory is None:
        _memory = VectorMemory()
    return _memory


def vector_db_store(content: str, metadata: Optional[dict] = None) -> dict:
    """Tool entry point: store a finding in long-term memory."""
    try:
        meta = metadata or {}
        return _get_memory().store(
            content,
            ticker=meta.get("ticker", ""),
            source_type=meta.get("source_type", "analysis"),
            date=meta.get("date"),
            confidence=float(meta.get("confidence", 0.8)),
            session=meta.get("session", ""),
            verified=bool(meta.get("verified", False)),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def vector_db_search(query: str, top_k: int = 5, filter: Optional[dict] = None) -> dict:
    """Tool entry point: semantic search over long-term memory."""
    try:
        return _get_memory().search(query, top_k=top_k, filter=filter)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    # Smoke test: store a couple of findings, then search.
    mem = VectorMemory()
    print("Starting count:", mem.count())
    mem.store("Microsoft's FY2025 revenue was $281.7B, up 18% YoY, driven by Azure cloud growth.",
              ticker="MSFT", source_type="financial_data_api", confidence=0.95)
    mem.store("Microsoft's profit margin is 39.3%, exceptionally high for its scale.",
              ticker="MSFT", source_type="analysis", confidence=0.9)
    print("After storing:", mem.count())
    print("\nSearch 'how profitable is Microsoft':")
    for r in mem.search("how profitable is Microsoft")["results"]:
        print(f"  [{r['similarity']}] ({r['ticker']}) {r['content'][:70]}...")