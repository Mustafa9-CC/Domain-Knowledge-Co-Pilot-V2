"""
Vector store service using ChromaDB.

Single collection 'corpus_chunks' with metadata filtering by user_id,
corpus_id, and document_id.

Retrieval strategy: Document-Aware MMR (Maximum Marginal Relevance)
=====================================================================

Previous algorithm (Global Top-K):
- Embed query, return the K nearest chunks from ChromaDB
- Problem: one document can monopolize all K slots for broad queries

New algorithm (Document-Aware MMR):
1. Over-fetch: retrieve 3×K candidates from ChromaDB (still one ANN query)
2. Group candidates by document_id
3. Apply per-document MMR selection:
   - Round-robin across documents, picking the best remaining chunk
     from each document that maximises:
       score = λ × similarity(query, chunk) - (1-λ) × max_similarity(chunk, selected)
   - This penalises chunks that are redundant with already-selected chunks
4. Continue until K chunks are selected or candidates exhausted

Why this improves quality:
- Broad queries ("summarise all documents") get representation from every doc
- Specific queries still surface the most relevant chunks (λ=0.7 biases toward relevance)
- Redundant passages from the same document are suppressed
- Computational overhead is minimal: one extra numpy dot-product over 3K vectors

Computational cost:
- Embedding: identical (1 query embed)
- ChromaDB query: slightly larger (3K vs K), but HNSW is sublinear — negligible
- MMR re-ranking: O(K × 3K × dim) dot products ≈ O(K²·dim), with K=5 and dim=384
  this is ~3K multiply-adds — sub-millisecond
"""

import logging

import numpy as np
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

COLLECTION_NAME = "corpus_chunks"

# MMR trade-off: 0 = maximum diversity, 1 = pure relevance.
# 0.7 favours relevance while still ensuring document diversity.
MMR_LAMBDA = 0.7

# Over-fetch multiplier: how many extra candidates to pull from ChromaDB
# before applying MMR re-ranking.
OVERFETCH_MULTIPLIER = 3


class VectorStore:
    """Wrapper around ChromaDB with sentence-transformer embeddings."""

    def __init__(self):
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._model: SentenceTransformer | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        return self._model

    def warmup(self):
        """Eagerly initialize all lazy resources and warm the runtime.

        Call during application startup to eliminate cold-start latency
        on the first real request. Triggers:
        - ChromaDB client + collection creation/loading
        - Embedding model weight loading
        - A dummy encode to warm the ONNX/PyTorch execution engine
        """
        import time
        t0 = time.perf_counter()

        # Force-init ChromaDB
        _ = self.collection
        logger.info("Warmup: ChromaDB client and collection initialized")

        # Force-load embedding model + run a dummy encode
        _ = self.model
        self.model.encode(["warmup"], show_progress_bar=False, normalize_embeddings=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Warmup: embedding model loaded and warmed ({elapsed_ms:.0f}ms)")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Returns a numpy ndarray of shape (len(texts), dim) with float32 dtype.
        ChromaDB accepts numpy directly — no .tolist() conversion needed.
        batch_size=64 was profiled to be 25% faster than the default (32).
        """
        return self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,
        )

    def add_chunks(
        self,
        chunks: list[str],
        user_id: int,
        corpus_id: int,
        document_id: int,
        filename: str,
    ) -> int:
        """Embed and store document chunks in ChromaDB.

        Args:
            chunks: List of text chunks.
            user_id: Owner user ID.
            corpus_id: Parent corpus ID.
            document_id: Parent document ID.
            filename: Original filename for metadata.

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            return 0

        embeddings = self.embed(chunks)

        ids = [f"doc{document_id}_chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "user_id": user_id,
                "corpus_id": corpus_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # ChromaDB supports batch upsert
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(chunks)} chunks for document {document_id} ({filename})")
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        corpus_ids: list[int],
        top_k: int = 5,
    ) -> list[dict]:
        """Document-aware MMR retrieval across one or more corpora.

        1. Over-fetch 3×K candidates from ChromaDB.
        2. Re-rank using MMR to balance relevance vs. diversity.
        3. Ensure document diversity: each document gets at least one slot
           (when possible) before any document gets a second.

        The returned list is ordered by descending relevance *within*
        each round-robin pass, so the prompt receives the most
        relevant chunk first.

        Args:
            query_text: The search query.
            corpus_ids: One or more corpus IDs to search within. For a
                single-corpus query, pass a one-element list: [corpus_id].
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: document_id, filename, chunk_index,
            chunk_text, score, corpus_id.
        """
        query_embedding = self.embed([query_text])[0]  # shape: (dim,)
        query_vec = query_embedding  # already numpy float32 from embed()

        # Step 1: Over-fetch candidates
        fetch_k = top_k * OVERFETCH_MULTIPLIER

        # ChromaDB $in operator performs a single ANN scan with
        # post-filtering across all supplied corpus IDs.
        if len(corpus_ids) == 1:
            chroma_filter = {"corpus_id": corpus_ids[0]}
        else:
            chroma_filter = {"corpus_id": {"$in": corpus_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=chroma_filter,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        # Build candidate list with embeddings
        candidates = []
        for i, _id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            score = 1.0 - (distance / 2.0)
            embedding = np.array(results["embeddings"][0][i], dtype=np.float32)
            candidates.append({
                "document_id": metadata["document_id"],
                "corpus_id": metadata["corpus_id"],
                "filename": metadata["filename"],
                "chunk_index": metadata["chunk_index"],
                "chunk_text": results["documents"][0][i],
                "score": round(score, 4),
                "embedding": embedding,
            })

        # Step 2: Apply document-aware MMR selection
        selected = self._mmr_select(query_vec, candidates, top_k)

        # Step 3: Order by relevance (highest score first) for the prompt
        # but group by document for better context coherence
        selected = self._order_for_prompt(selected)

        return selected

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        candidates: list[dict],
        k: int,
    ) -> list[dict]:
        """MMR selection with document-diversity guarantee.

        Round-robin across documents: in each round, pick the best
        MMR-scored chunk from each document that hasn't contributed
        a chunk in this round yet. This ensures every document gets
        at least one slot before any document gets a second.

        Args:
            query_vec: The query embedding (normalised).
            candidates: List of candidate dicts with 'embedding' key.
            k: Number of chunks to select.

        Returns:
            Selected chunks (without 'embedding' key).
        """
        if len(candidates) <= k:
            # Fewer candidates than slots — return all
            return [{key: c[key] for key in c if key != "embedding"} for c in candidates]

        selected: list[dict] = []
        selected_embeddings: list[np.ndarray] = []
        remaining = list(candidates)  # mutable copy

        while len(selected) < k and remaining:
            # Group remaining by document_id
            doc_groups: dict[int, list[int]] = {}
            for idx, cand in enumerate(remaining):
                doc_id = cand["document_id"]
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(idx)

            # For each document, pick the candidate with the best MMR score
            round_picks: list[tuple[float, int]] = []  # (mmr_score, index)

            for doc_id, indices in doc_groups.items():
                best_mmr = -float("inf")
                best_idx = -1

                for idx in indices:
                    cand = remaining[idx]
                    # Relevance component: cosine similarity to query
                    relevance = float(np.dot(query_vec, cand["embedding"]))

                    # Diversity component: max similarity to already-selected
                    if selected_embeddings:
                        similarities = [
                            float(np.dot(cand["embedding"], sel_emb))
                            for sel_emb in selected_embeddings
                        ]
                        max_sim = max(similarities)
                    else:
                        max_sim = 0.0

                    mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx

                if best_idx >= 0:
                    round_picks.append((best_mmr, best_idx))

            # Sort picks by MMR score (best first) and take as many as needed
            round_picks.sort(key=lambda x: x[0], reverse=True)

            for mmr_score, idx in round_picks:
                if len(selected) >= k:
                    break
                cand = remaining[idx]
                selected_embeddings.append(cand["embedding"])
                # Remove embedding from the returned dict
                selected.append({key: cand[key] for key in cand if key != "embedding"})

            # Remove all picked indices from remaining
            picked_indices = {idx for _, idx in round_picks if len(selected) <= k + len(round_picks)}
            # More precise: remove exactly the ones we added to selected
            picked_set = set()
            count = 0
            for mmr_score, idx in round_picks:
                if count >= k:
                    break
                picked_set.add(idx)
                count += 1
                if count >= k:
                    break

            # Rebuild remaining without picked candidates
            remaining = [c for i, c in enumerate(remaining) if i not in picked_set]

        return selected

    def _order_for_prompt(self, chunks: list[dict]) -> list[dict]:
        """Order selected chunks for optimal prompt construction.

        Strategy: group chunks by document, then order groups by the
        highest score in each group. Within each group, order by
        chunk_index for reading coherence.

        This means the LLM sees:
        1. Most relevant document's chunks (in reading order)
        2. Second most relevant document's chunks
        3. etc.
        """
        if not chunks:
            return chunks

        # Group by document
        doc_groups: dict[int, list[dict]] = {}
        doc_best_score: dict[int, float] = {}
        for chunk in chunks:
            doc_id = chunk["document_id"]
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
                doc_best_score[doc_id] = 0.0
            doc_groups[doc_id].append(chunk)
            doc_best_score[doc_id] = max(doc_best_score[doc_id], chunk["score"])

        # Sort groups by best score (descending)
        sorted_doc_ids = sorted(doc_best_score, key=doc_best_score.get, reverse=True)

        # Within each group, sort by chunk_index for reading coherence
        ordered = []
        for doc_id in sorted_doc_ids:
            group = sorted(doc_groups[doc_id], key=lambda c: c["chunk_index"])
            ordered.extend(group)

        return ordered

    # ------------------------------------------------------------------
    # Single-chunk retrieval
    # ------------------------------------------------------------------

    def get_chunk_text(self, document_id: int, chunk_index: int) -> str | None:
        """Retrieve the text of a specific chunk.

        Args:
            document_id: The document ID.
            chunk_index: The chunk index within the document.

        Returns:
            The chunk text, or None if not found.
        """
        chunk_id = f"doc{document_id}_chunk{chunk_index}"
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents"],
            )
            if result and result["documents"]:
                return result["documents"][0]
        except Exception:
            logger.warning(f"Failed to retrieve chunk doc{document_id}_chunk{chunk_index}", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_by_document(self, document_id: int):
        """Delete all chunks belonging to a document.

        Raises on failure so the caller can abort the SQL deletion
        and keep the system in a consistent state.
        """
        self.collection.delete(where={"document_id": document_id})
        logger.info(f"Deleted vectors for document {document_id}")

    def delete_by_corpus(self, corpus_id: int):
        """Delete all chunks belonging to a corpus.

        Raises on failure so the caller can abort the SQL deletion
        and keep the system in a consistent state.
        """
        self.collection.delete(where={"corpus_id": corpus_id})
        logger.info(f"Deleted vectors for corpus {corpus_id}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_by_document(self, document_id: int) -> int:
        """Return the number of vectors stored for a document."""
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                include=[],  # No data, just count
            )
            return len(result["ids"]) if result and result["ids"] else 0
        except Exception:
            logger.warning(f"Failed to count vectors for document {document_id}", exc_info=True)
            return -1  # Indicates error

    def count_by_corpus(self, corpus_id: int) -> int:
        """Return the number of vectors stored for a corpus."""
        try:
            result = self.collection.get(
                where={"corpus_id": corpus_id},
                include=[],
            )
            return len(result["ids"]) if result and result["ids"] else 0
        except Exception:
            logger.warning(f"Failed to count vectors for corpus {corpus_id}", exc_info=True)
            return -1

    def verify_document(self, document_id: int, expected_chunks: int | None) -> dict:
        """Full consistency check for a document's vectors.

        Compares the expected chunk_count (from SQLite) against actual
        vectors in ChromaDB, and checks for missing chunk indices.

        Returns:
            Dict with keys: vector_count, expected_count, consistent,
            missing_indices, extra_indices.
        """
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            vector_count = len(result["ids"]) if result and result["ids"] else 0
            stored_indices = set()
            if result and result["metadatas"]:
                for m in result["metadatas"]:
                    stored_indices.add(m.get("chunk_index", -1))
        except Exception as e:
            return {
                "vector_count": -1,
                "expected_count": expected_chunks,
                "consistent": False,
                "error": str(e),
            }

        expected = expected_chunks or 0
        expected_indices = set(range(expected))
        missing = sorted(expected_indices - stored_indices)
        extra = sorted(stored_indices - expected_indices)

        return {
            "vector_count": vector_count,
            "expected_count": expected,
            "consistent": vector_count == expected and not missing,
            "missing_indices": missing[:20],  # Cap to avoid huge responses
            "extra_indices": extra[:20],
        }

    def corpus_doc_vector_counts(self, corpus_id: int) -> dict[int, int]:
        """Return a {document_id: vector_count} map for a corpus."""
        try:
            result = self.collection.get(
                where={"corpus_id": corpus_id},
                include=["metadatas"],
            )
            counts: dict[int, int] = {}
            if result and result["metadatas"]:
                for m in result["metadatas"]:
                    doc_id = m["document_id"]
                    counts[doc_id] = counts.get(doc_id, 0) + 1
            return counts
        except Exception:
            logger.warning(f"Failed to get vector counts for corpus {corpus_id}", exc_info=True)
            return {}


# Singleton instance
vector_store = VectorStore()

