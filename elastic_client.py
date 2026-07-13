"""
Elasticsearch client - Index and search documents
"""

import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import config
import cache_manager


# Elasticsearch index mapping
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "document_id": {
                "type": "keyword"
            },
            "text": {
                "type": "text",
                "analyzer": "arabic"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "story_titel": {"type": "text", "analyzer": "arabic"},
                    "story_keywords": {"type": "keyword"},
                    "story_category": {"type": "keyword"},
                    "story_sammary": {"type": "text", "analyzer": "arabic"},
                    "story_entities": {
                        "type": "nested",
                        "properties": {
                            "entity_value": {"type": "keyword"},
                            "entity_type": {"type": "keyword"}
                        }
                    }
                }
            },
            "dense_vector": {
                "type": "dense_vector",
                "dims": config.VECTOR_DIMS,
                "index": True,
                "similarity": "cosine"
            },
            "sparse_vector": {
                "type": "sparse_vector"
            },
            "indexed_at": {
                "type": "date"
            }
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": config.ES_REFRESH_INTERVAL
    }
}


class ESClient:
    """Elasticsearch operations"""
    
    def __init__(self):
        self.es = Elasticsearch(config.ES_HOST)
        self._ensure_index()
    
    def _ensure_index(self):
        """Create index if not exists or update settings"""
        if not self.es.indices.exists(index=config.ES_INDEX):
            self.es.indices.create(index=config.ES_INDEX, body=INDEX_MAPPING)
            print(f"[ES] Created index: {config.ES_INDEX}")
        else:
            # Update settings for existing index (e.g. refresh_interval)
            try:
                self.es.indices.put_settings(
                    index=config.ES_INDEX,
                    body={"index": {"refresh_interval": config.ES_REFRESH_INTERVAL}}
                )
                print(f"[ES] Updated index settings: {config.ES_INDEX}")
            except Exception as e:
                print(f"[ES] Failed to update settings: {e}")
            print(f"[ES] Index exists: {config.ES_INDEX}")
    
    def index_document(self, doc_id, text, metadata, vectors):
        """
        Index a single document
        
        doc_id: unique document ID
        text: original Arabic text
        metadata: extracted metadata dict
        vectors: {dense: [...], sparse: {...}}
        """
        doc = {
            "document_id": doc_id,
            "text": text,
            "metadata": metadata,
            "dense_vector": vectors["dense"],
            "sparse_vector": vectors["sparse"],
            "indexed_at": datetime.utcnow().isoformat()
        }
        
        self.es.index(index=config.ES_INDEX, id=doc_id, document=doc)
        print(f"[ES] Indexed: {doc_id}")
    
    def bulk_index(self, documents):
        """
        Bulk index multiple documents
        
        documents: list of {doc_id, text, metadata, vectors}
        """
        actions = []
        for doc in documents:
            actions.append({
                "_index": config.ES_INDEX,
                "_id": doc["doc_id"],
                "_source": {
                    "document_id": doc["doc_id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "dense_vector": doc["vectors"]["dense"],
                    "sparse_vector": doc["vectors"]["sparse"],
                    "indexed_at": datetime.utcnow().isoformat()
                }
            })
        
        success, failed = bulk(self.es, actions, raise_on_error=False)
        print(f"[ES] Bulk indexed: {success} success, {len(failed)} failed")
        return success, len(failed)
    
    def search(self, query_text, dense_vector=None, sparse_vector=None, size=10, use_bm25=True, use_frr=True):
        """
        Hybrid search: BM25 + Dense k-NN + Sparse Vector (manual RRF fusion)
        Uses Python-side RRF instead of ES-native RRF to avoid license requirements.
        Queries run in parallel for speed.
        """
        RRF_K = 60
        candidate_size = config.ES_CANDIDATE_SIZE

        def _exec_bm25():
            body = {
                "query": {
                    "match": {
                        "text": query_text
                    }
                }
            }
            return self.es.search(index=config.ES_INDEX, body=body, size=candidate_size)

        def _exec_knn():
            vec = dense_vector.tolist() if hasattr(dense_vector, 'tolist') else list(dense_vector)
            body = {
                "knn": {
                    "field": "dense_vector",
                    "query_vector": vec,
                    "k": candidate_size,
                    "num_candidates": candidate_size
                }
            }
            return self.es.search(index=config.ES_INDEX, body=body, size=candidate_size)

        def _exec_sparse():
            body = {
                "query": {
                    "sparse_vector": {
                        "field": "sparse_vector",
                        "query_vector": sparse_vector
                    }
                }
            }
            return self.es.search(index=config.ES_INDEX, body=body, size=candidate_size)

        tasks = []
        if use_bm25:
            tasks.append((_exec_bm25, "bm25"))
        if dense_vector is not None:
            tasks.append((_exec_knn, "knn"))
        if sparse_vector is not None:
            try:
                sv_len = len(sparse_vector)
            except TypeError:
                sv_len = 1
            if sv_len > 0:
                tasks.append((_exec_sparse, "sparse"))

        if not tasks:
            return []

        responses = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for fn, name in tasks}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    responses[name] = future.result()
                except Exception as e:
                    print(f"[ES] Error in {name} search: {e}")

        all_results = {}
        for name, resp in responses.items():
            for rank, hit in enumerate(resp["hits"]["hits"]):
                doc_id = hit["_source"]["document_id"]
                if doc_id not in all_results:
                    # Initialize scores for all requested algorithms
                    initial_scores = {}
                    if use_bm25: initial_scores["bm25"] = 0.0
                    if dense_vector is not None: initial_scores["knn"] = 0.0
                    if sparse_vector is not None: initial_scores["sparse"] = 0.0
                    
                    all_results[doc_id] = {
                        "data": hit["_source"],
                        "rrf_score": 0.0,
                        "scores": initial_scores
                    }
                if use_frr:
                    # Apply weighted RRF
                    weights = {
                        "bm25": config.ES_RRF_WEIGHT_BM25,
                        "knn": config.ES_RRF_WEIGHT_DENSE,
                        "sparse": config.ES_RRF_WEIGHT_SPARSE
                    }
                    weight = weights.get(name, 1.0)
                    all_results[doc_id]["rrf_score"] += weight * (1.0 / (rank + 1 + RRF_K))
                all_results[doc_id]["scores"][name] = hit["_score"]

        if use_frr:
            sorted_results = sorted(
                all_results.items(),
                key=lambda x: x[1]["rrf_score"],
                reverse=True
            )[:size]
        else:
            sorted_results = sorted(
                all_results.items(),
                key=lambda x: max(x[1]["scores"].values()),
                reverse=True
            )[:size]

        results = []
        for doc_id, entry in sorted_results:
            data = entry["data"]
            results.append({
                "score": round(entry["rrf_score"], 6) if use_frr else None,
                "individual_scores": entry["scores"],
                "document_id": doc_id,
                "text": data["text"][:500] + "...",
                "metadata": data["metadata"],
                "indexed_at": data.get("indexed_at", "")
            })

        return results
    
    def search_bm25_only(self, query_text, size=10):
        """BM25 text search only"""
        query_body = {
            "query": {
                "match": {
                    "text": query_text
                }
            }
        }
        response = self.es.search(index=config.ES_INDEX, body=query_body, size=size)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "score": hit["_score"],
                "document_id": hit["_source"]["document_id"],
                "text": hit["_source"]["text"][:500] + "...",
                "metadata": hit["_source"]["metadata"]
            })
        return results
    
    def search_knn_only(self, dense_vector, size=10):
        """Dense k-NN search only"""
        query_body = {
            "knn": {
                "field": "dense_vector",
                "query_vector": dense_vector.tolist() if hasattr(dense_vector, 'tolist') else dense_vector,
                "k": size,
                "num_candidates": 100
            }
        }
        response = self.es.search(index=config.ES_INDEX, body=query_body, size=size)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "score": hit["_score"],
                "document_id": hit["_source"]["document_id"],
                "text": hit["_source"]["text"][:500] + "...",
                "metadata": hit["_source"]["metadata"]
            })
        return results
    
    def get_document(self, doc_id):
        """Get full document by ID"""
        try:
            response = self.es.get(index=config.ES_INDEX, id=doc_id)
            return response["_source"]
        except Exception:
            return None
    
    def delete_index(self):
        """Delete the index (use with caution)"""
        if self.es.indices.exists(index=config.ES_INDEX):
            self.es.indices.delete(index=config.ES_INDEX)
            print(f"[ES] Deleted index: {config.ES_INDEX}")
    
    def get_stats(self):
        """Get index statistics"""
        if self.es.indices.exists(index=config.ES_INDEX):
            stats = self.es.indices.stats(index=config.ES_INDEX)
            count = self.es.count(index=config.ES_INDEX)["count"]
            return {
                "document_count": count,
                "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"]
            }
        return {"document_count": 0, "size_bytes": 0}
    
    def reindex_from_cache(self):
        """
        Reindex all documents from cache
        Useful after clearing ES or changing mapping
        """
        cached_files = cache_manager.get_all_cached_files()
        print(f"[ES] Reindexing {len(cached_files)} cached documents")
        
        documents = []
        for file_id in cached_files:
            cache_data = cache_manager.load_cache(file_id)
            
            if cache_data.get("metadata") and cache_data.get("vectors"):
                # Use corrected text if available, else original
                text = cache_data.get("corrected_text") or cache_data.get("original_text", "")
                
                documents.append({
                    "doc_id": file_id,
                    "text": text,
                    "metadata": cache_data["metadata"],
                    "vectors": cache_data["vectors"]
                })
        
        if documents:
            return self.bulk_index(documents)
        return 0, 0
    
    def _find_original_file(self, file_id):
        """Find original JSON file by ID"""
        import os
        for f in os.listdir(config.WATCH_FOLDER):
            if f.startswith(file_id) and f.endswith(".json"):
                return os.path.join(config.WATCH_FOLDER, f)
        return None


# Global instance
_es_client = None


def get_es_client():
    global _es_client
    if _es_client is None:
        _es_client = ESClient()
    return _es_client
