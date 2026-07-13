"""
Configuration - Edit this file to change settings
"""

# Paths
PROJECT_DIR = "/data/ArabicLLM/elastic_rag"
WATCH_FOLDER = f"{PROJECT_DIR}/watch_folder"
CACHE_DIR = f"{PROJECT_DIR}/cache"  # Unified cache directory

# Elasticsearch
ES_HOST = "http://localhost:9200"
ES_INDEX = "arabic_docs"
ES_CANDIDATE_SIZE = 100  # Number of documents to fetch per algorithm
ES_REFRESH_INTERVAL = "1s" # Set to "1s" to enable auto-refresh (standard for dev)
ES_RRF_WEIGHT_BM25 = 0.6
ES_RRF_WEIGHT_DENSE = 0.3
ES_RRF_WEIGHT_SPARSE = 0.1

# Models
DENSE_MODEL = "microsoft/harrier-oss-v1-270m"
SPARSE_MODEL = "Omartificial-Intelligence-Space/inference-free-splade-distilbert-base-Arabic-cased-nq"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_ADAPTER = f"{PROJECT_DIR}/model"  # Set path like: f"{PROJECT_DIR}/model"

# Text Correction (T5)
ENABLE_TEXT_CORRECTION = False  # Set to True to enable
WAIT_FOR_CORRECTION = False     # If True, metadata extraction waits for corrected text
CORRECTION_WAIT_TIMEOUT = 30    # Max seconds to wait for correction
TEXT_CORRECTION_MODEL = "SuperSl6/Arabic-Text-Correction"

# Metadata extraction settings
SUMMARY_MIN_POINTS = 3
SUMMARY_MAX_POINTS = 7

# Processing
BATCH_SIZE = 7
MAX_WORKERS = 10
MAX_CORRECTION_WORKERS = 6  # Separate workers for text correction
VECTOR_DIMS = 640

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
USE_NGROK = True

# Ngrok
NGROK_AUTH_TOKEN = '33mْْْْْXXXXXXXXXXXXXXXXXXXXXXXXXX' 
