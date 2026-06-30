import logging
import time
import string
from typing import Dict, List, Any
from rank_bm25 import BM25Okapi

from backend.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# TODO: Future optimization: For large corpora, BM25 indexing in memory becomes a bottleneck.
# Consider using an external search engine (e.g. Elasticsearch, OpenSearch) or a disk-backed inverted index.

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with punctuation stripping and lowercasing.
    
    TODO: Future optimization: Use a more advanced tokenizer (like spacy or NLTK) 
    for stemming and stop-word removal to improve BM25 recall.
    """
    if not text:
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class BM25IndexManager:
    def __init__(self):
        # Maps corpus_id -> BM25Okapi instance
        self.indices: Dict[int, BM25Okapi] = {}
        # Maps corpus_id -> list of chunk dicts (maintaining same index order as BM25Okapi documents)
        self.chunk_metadata: Dict[int, List[Dict[str, Any]]] = {}

    def warmup(self):
        """Build BM25 indices for all corpora from ChromaDB during startup."""
        t0 = time.perf_counter()
        
        from backend.database import SessionLocal
        from backend.models import Corpus
        
        db = SessionLocal()
        try:
            corpora = db.query(Corpus).all()
            corpus_ids = [c.id for c in corpora]
        except Exception as e:
            logger.error(f"Failed to fetch corpora during BM25 warmup: {e}")
            corpus_ids = []
        finally:
            db.close()

        count = 0
        for cid in corpus_ids:
            try:
                self.build_corpus(cid)
                count += 1
            except Exception as e:
                logger.error(f"Failed to warmup BM25 for corpus {cid}: {e}")

        t1 = time.perf_counter()
        logger.info(f"BM25 warmup completed in {t1 - t0:.3f}s for {count} corpora.")

    def build_corpus(self, corpus_id: int):
        """Fetch all chunks for a corpus from ChromaDB and rebuild the BM25 index."""
        t0 = time.perf_counter()
        
        try:
            # Query vector_store for all chunks in this corpus
            results = vector_store.collection.get(
                where={"corpus_id": corpus_id},
                include=["documents", "metadatas"]
            )
            
            if not results or not results["ids"]:
                # Empty corpus safeguard
                self.indices.pop(corpus_id, None)
                self.chunk_metadata.pop(corpus_id, None)
                logger.info(f"BM25 build_corpus({corpus_id}): Corpus is empty. Removed from index.")
                return

            documents = results["documents"]
            metadatas = results["metadatas"]
            
            tokenized_corpus = [tokenize(doc) for doc in documents]
            
            # Store BM25 object
            bm25 = BM25Okapi(tokenized_corpus)
            self.indices[corpus_id] = bm25
            
            # Reconstruct chunk metadata to return on query
            chunks = []
            for i, metadata in enumerate(metadatas):
                chunks.append({
                    "document_id": metadata["document_id"],
                    "corpus_id": metadata["corpus_id"],
                    "filename": metadata["filename"],
                    "chunk_index": metadata["chunk_index"],
                    "chunk_text": documents[i],
                })
            self.chunk_metadata[corpus_id] = chunks
            
            t1 = time.perf_counter()
            logger.info(f"BM25 build_corpus({corpus_id}) completed in {t1 - t0:.3f}s. Indexed {len(chunks)} chunks.")
            
        except Exception as e:
            logger.exception(f"BM25 build_corpus({corpus_id}) failed: {e}")
            # Safeguard: Do not leave corrupted state
            self.indices.pop(corpus_id, None)
            self.chunk_metadata.pop(corpus_id, None)
            raise

    def query(self, query_text: str, corpus_ids: List[int], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query BM25 index across one or more corpora and return top K chunks."""
        if not query_text or not query_text.strip():
            # Safeguard: empty query
            return []
            
        tokenized_query = tokenize(query_text)
        
        all_results = []
        for cid in corpus_ids:
            if cid not in self.indices:
                # Missing corpus safeguard
                continue
                
            bm25 = self.indices[cid]
            chunks = self.chunk_metadata[cid]
            
            scores = bm25.get_scores(tokenized_query)
            
            # Get top_k for this corpus
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0:
                    chunk_info = chunks[idx].copy()
                    chunk_info["score"] = score
                    all_results.append(chunk_info)
                    
        # Sort across all requested corpora by score descending and take top_k overall
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def add_document(self, corpus_id: int, document_id: int):
        """Update BM25 index when a document is added.
        
        TODO: Future optimization: Implement incremental BM25 updates instead of full corpus rebuild.
        """
        t0 = time.perf_counter()
        logger.info(f"BM25 add_document(corpus={corpus_id}, doc={document_id}) triggered.")
        self.build_corpus(corpus_id)
        t1 = time.perf_counter()
        logger.info(f"BM25 add_document complete in {t1 - t0:.3f}s")

    def remove_document(self, corpus_id: int, document_id: int):
        """Update BM25 index when a document is removed."""
        t0 = time.perf_counter()
        logger.info(f"BM25 remove_document(corpus={corpus_id}, doc={document_id}) triggered.")
        self.build_corpus(corpus_id)
        t1 = time.perf_counter()
        logger.info(f"BM25 remove_document complete in {t1 - t0:.3f}s")

    def remove_corpus(self, corpus_id: int):
        """Remove a corpus entirely from the BM25 index."""
        if corpus_id in self.indices:
            del self.indices[corpus_id]
        if corpus_id in self.chunk_metadata:
            del self.chunk_metadata[corpus_id]
        logger.info(f"BM25 remove_corpus({corpus_id}) completed.")

bm25_index = BM25IndexManager()
