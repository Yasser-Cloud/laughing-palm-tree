"""
Document processor - Multi-stage pipeline with separate queues
"""

import json
import time
import threading
from queue import Queue, Empty
import logging

import config
import cache_manager as cache
from models import get_dense_model, get_sparse_model, get_qwen_model, get_text_corrector
from elastic_client import get_es_client

logger = logging.getLogger("processor")


class DocumentProcessor:
    """
    Process documents through multi-stage pipeline:
    1. Text Correction (optional, separate queue)
    2. Metadata Extraction
    3. Vector Generation
    4. Index to Elasticsearch
    """
    
    def __init__(self):
        self.main_queue = Queue()
        self.correction_queue = Queue()
        self.workers = []
        self.correction_workers = []
        self.running = False
    
    def add_file(self, file_path):
        """Add file to processing queue"""
        self.main_queue.put(file_path)
        if config.ENABLE_TEXT_CORRECTION:
            self.correction_queue.put(file_path)
    
    def start(self, num_workers=1, num_correction_workers=1):
        """Start worker threads"""
        self.running = True
        
        # Start text correction workers (separate queue)
        if config.ENABLE_TEXT_CORRECTION:
            for i in range(num_correction_workers):
                worker = threading.Thread(target=self._correction_worker, args=(i,), daemon=True)
                worker.start()
                self.correction_workers.append(worker)
        
        # Start main processing workers
        for i in range(num_workers):
            worker = threading.Thread(target=self._main_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop all workers"""
        self.running = False
        for worker in self.workers + self.correction_workers:
            worker.join(timeout=5)
    
    def _correction_worker(self, worker_id):
        """Separate worker for text correction (non-blocking)"""
        corrector = None
        
        while self.running:
            file_path = None
            try:
                file_path = self.correction_queue.get(timeout=1)
                
                if corrector is None:
                    corrector = get_text_corrector()
                
                self._correct_text(file_path, corrector)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Correction error: {e}")
            finally:
                if file_path is not None:
                    self.correction_queue.task_done()
    
    def _correct_text(self, file_path, corrector):
        """Stage 1: Correct text (separate queue)"""
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        
        text = data.get("asr_ocr_text", "")
        if not text:
            return
        
        file_id = cache.get_file_id(file_path)
        cache_data = cache.load_cache(file_id)
        
        # Skip if already corrected
        if cache_data.get("corrected_text"):
            return
        
        # Save original text
        if not cache_data.get("original_text"):
            cache.update_cache(file_id, "original_text", text)
        
        # Correct text
        corrected = corrector.correct(text)
        cache.update_cache(file_id, "corrected_text", corrected)
    
    def _main_worker(self, worker_id):
        """Main worker for metadata extraction and indexing"""
        dense_model = None
        sparse_model = None
        qwen_model = None
        es_client = None
        
        while self.running:
            file_path = None
            try:
                file_path = self.main_queue.get(timeout=1)
                
                if dense_model is None:
                    dense_model = get_dense_model()
                if sparse_model is None:
                    sparse_model = get_sparse_model()
                if qwen_model is None:
                    qwen_model = get_qwen_model()
                if es_client is None:
                    es_client = get_es_client()
                
                self._process_file(file_path, dense_model, sparse_model, qwen_model, es_client)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if file_path is not None:
                    self.main_queue.task_done()
    
    def _process_file(self, file_path, dense_model, sparse_model, qwen_model, es_client):
        """Process file through main pipeline"""
        
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        
        text = data.get("asr_ocr_text", "")
        if not text:
            return
        
        file_id = cache.get_file_id(file_path)
        cache_data = cache.load_cache(file_id)
        
        # Stage 1: Save original text
        if not cache_data.get("original_text"):
            cache.update_cache(file_id, "original_text", text)
        
        # Stage 2: Wait for corrected text (if enabled)
        if config.ENABLE_TEXT_CORRECTION and config.WAIT_FOR_CORRECTION:
            waited = 0
            while waited < config.CORRECTION_WAIT_TIMEOUT:
                cache_data = cache.load_cache(file_id)
                if cache_data.get("corrected_text"):
                    break
                time.sleep(1)
                waited += 1
        
        # Stage 3: Get text for processing (corrected or original)
        cache_data = cache.load_cache(file_id)  # Refresh cache data
        process_text = cache.get_text_for_processing(file_id)
        
        # Stage 4: Extract metadata
        if not cache_data.get("metadata"):
            metadata = qwen_model.extract(process_text)
            cache.update_cache(file_id, "metadata", metadata)
        else:
            metadata = cache_data["metadata"]
        
        # Stage 5: Generate vectors
        if not cache_data.get("vectors"):
            combined_text = self._create_combined_text(process_text, metadata)
            
            dense_vec = dense_model.encode(combined_text)[0]
            sparse_vec = sparse_model.encode(combined_text)[0]
            
            vectors = {
                "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec),
                "sparse": sparse_vec
            }
            cache.update_cache(file_id, "vectors", vectors)
        else:
            vectors = cache_data["vectors"]
        
        # Stage 6: Index to Elasticsearch
        es_client.index_document(file_id, process_text, metadata, vectors)
    
    def _create_combined_text(self, text, metadata):
        """Combine metadata for embedding"""
        parts = []
        
        if metadata.get("story_titel"):
            parts.append(f"العنوان: {metadata['story_titel']}")
        
        if metadata.get("story_keywords"):
            if isinstance(metadata["story_keywords"], list):
                parts.append(f"الكلمات المفتاحية: {', '.join(str(kw) for kw in metadata['story_keywords'])}")
        
        if metadata.get("story_sammary"):
            if isinstance(metadata["story_sammary"], list):
                parts.append(f"الملخص: {' '.join(str(s) for s in metadata['story_sammary'])}")
        
        if metadata.get("story_entities"):
            entities = []
            for e in metadata["story_entities"]:
                if isinstance(e, dict):
                    entities.append(e.get("entity_value", ""))
                elif isinstance(e, str):
                    try:
                        parsed = json.loads(e)
                        entities.append(parsed.get("entity_value", ""))
                    except:
                        pass
            if entities:
                parts.append(f"الكيانات: {', '.join(entities)}")
        
        return " ".join(parts) if parts else text[:500]
    
    def get_queue_sizes(self):
        """Get queue sizes"""
        return {
            "main_queue": self.main_queue.qsize(),
            "correction_queue": self.correction_queue.qsize()
        }


_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
