"""
File watcher - Monitor folder for new JSON files using watchdog
"""

import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import config
from processor import get_processor

logger = logging.getLogger("watcher")


class JSONFileHandler(FileSystemEventHandler):
    """Handles file system events for JSON files"""
    
    def __init__(self, processor):
        self.processor = processor
        self.processed_files = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".json"):
            self._process_file(event.src_path)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".json"):
            file_id = os.path.splitext(os.path.basename(event.src_path))[0]
            if file_id not in self.processed_files:
                self._process_file(event.src_path)
    
    def _process_file(self, file_path):
        time.sleep(0.5)
        
        if os.path.exists(file_path):
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            self.processed_files.add(file_id)
            self.processor.add_file(file_path)
    
    def scan_existing_files(self):
        """High-performance lazy scan of existing files"""
        if not os.path.exists(config.WATCH_FOLDER):
            return
        
        print(f"[Watcher] Scanning {config.WATCH_FOLDER} for new files...")
        count = 0
        skipped = 0
        
        for root, _, files in os.walk(config.WATCH_FOLDER):
            for f in files:
                if not f.endswith(".json"):
                    continue
                    
                file_path = os.path.join(root, f)
                file_id = os.path.splitext(f)[0]
                
                # Instant check: Is it in cache?
                cache_path = os.path.join(config.CACHE_DIR, f"{file_id}.json")
                if os.path.exists(cache_path):
                    self.processed_files.add(file_id)
                    skipped += 1
                    continue
                
                if file_id not in self.processed_files:
                    self.processed_files.add(file_id)
                    self.processor.add_file(file_path)
                    count += 1
        
        print(f"[Watcher] Scan complete. Added: {count}, Already Cached: {skipped}")


class FolderWatcher:
    """Watches a folder for new JSON files and processes them"""
    
    def __init__(self):
        self.observer = Observer()
        self.processor = get_processor()
        self.handler = JSONFileHandler(self.processor)
    
    def start(self):
        os.makedirs(config.WATCH_FOLDER, exist_ok=True)
        
        num_correction = config.MAX_CORRECTION_WORKERS if config.ENABLE_TEXT_CORRECTION else 0
        self.processor.start(
            num_workers=config.MAX_WORKERS,
            num_correction_workers=num_correction
        )
        
        self.handler.scan_existing_files()
        
        self.observer.schedule(self.handler, config.WATCH_FOLDER, recursive=True)
        self.observer.start()
    
    def stop(self):
        self.observer.stop()
        self.observer.join()
        self.processor.stop()
    
    def get_status(self):
        queue_sizes = self.processor.get_queue_sizes()
        return {
            "watch_folder": config.WATCH_FOLDER,
            "main_queue": queue_sizes.get("main_queue", 0),
            "correction_queue": queue_sizes.get("correction_queue", 0),
            "processed_files": len(self.handler.processed_files),
            "text_correction_enabled": config.ENABLE_TEXT_CORRECTION
        }


_watcher = None


def get_watcher():
    global _watcher
    if _watcher is None:
        _watcher = FolderWatcher()
    return _watcher
