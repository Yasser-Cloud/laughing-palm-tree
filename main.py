"""
Main entry point - Start all components
"""

import sys
import threading

import config
from watcher import get_watcher
from api import run_server


def main():
    """Start the application"""
    print("\n" + "="*60)
    print("Arabic Document Search - Elastic RAG")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Watch folder: {config.WATCH_FOLDER}")
    print(f"  Cache dir: {config.CACHE_DIR}")
    print(f"  Elasticsearch: {config.ES_HOST}")
    print(f"  Index: {config.ES_INDEX}")
    print(f"  Main workers: {config.MAX_WORKERS}")
    print(f"  API port: {config.API_PORT}")
    print(f"\nText Correction:")
    print(f"  Enabled: {config.ENABLE_TEXT_CORRECTION}")
    if config.ENABLE_TEXT_CORRECTION:
        print(f"  Wait for correction: {config.WAIT_FOR_CORRECTION}")
        print(f"  Correction workers: {config.MAX_CORRECTION_WORKERS}")
    print(f"\nModels:")
    print(f"  Dense: {config.DENSE_MODEL}")
    print(f"  Sparse: {config.SPARSE_MODEL}")
    print(f"  LLM: {config.QWEN_MODEL}")
    print(f"  Adapter: {config.QWEN_ADAPTER}")
    print("="*60 + "\n")
    
    # Start watcher (will also start processor)
    print("[Main] Starting file watcher...")
    watcher = get_watcher()
    watcher.start()
    
    # Run API server (blocking)
    print("[Main] Starting API server...")
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        watcher.stop()
        print("[Main] Goodbye!")


if __name__ == "__main__":
    main()
