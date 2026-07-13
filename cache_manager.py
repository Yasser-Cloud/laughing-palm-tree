"""
Cache manager - Single file per document with progressive data
"""

import os
import json
import config


def get_file_id(file_path):
    """Extract file ID from path (without extension)"""
    return os.path.splitext(os.path.basename(file_path))[0]


def get_cache_path(file_id):
    """Path to unified cache file"""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, f"{file_id}.json")


def save_json(path, data):
    """Save data to JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    """Load data from JSON file"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def cache_exists(file_id):
    """Check if cache file exists"""
    return os.path.exists(get_cache_path(file_id))


def load_cache(file_id):
    """Load cache file, return empty dict if not exists"""
    path = get_cache_path(file_id)
    data = load_json(path)
    if data is None:
        data = {}
    return data


def save_cache(file_id, data):
    """Save cache file"""
    path = get_cache_path(file_id)
    save_json(path, data)
    print(f"[Cache] Saved: {file_id}")


def update_cache(file_id, key, value):
    """Update specific key in cache (append mode)"""
    data = load_cache(file_id)
    data[key] = value
    save_cache(file_id, data)


def has_original_text(file_id):
    return "original_text" in load_cache(file_id)


def has_corrected_text(file_id):
    return "corrected_text" in load_cache(file_id)


def has_metadata(file_id):
    return "metadata" in load_cache(file_id)


def has_vectors(file_id):
    return "vectors" in load_cache(file_id)


def save_original_text(file_id, text):
    """Stage 1: Save original text"""
    update_cache(file_id, "original_text", text)


def save_corrected_text(file_id, text):
    """Stage 2: Save corrected text"""
    update_cache(file_id, "corrected_text", text)


def save_metadata(file_id, metadata):
    """Stage 3: Save extracted metadata"""
    update_cache(file_id, "metadata", metadata)


def save_vectors(file_id, vectors):
    """Stage 4: Save vectors"""
    update_cache(file_id, "vectors", vectors)


def get_text_for_processing(file_id):
    """Get text for processing (corrected if available, else original)"""
    data = load_cache(file_id)
    if config.ENABLE_TEXT_CORRECTION and data.get("corrected_text"):
        return data["corrected_text"]
    return data.get("original_text", "")


def get_all_cached_files():
    """Get list of all cached files"""
    files = []
    if os.path.exists(config.CACHE_DIR):
        for f in os.listdir(config.CACHE_DIR):
            if f.endswith(".json"):
                files.append(f[:-5])
    return files


def get_processing_status(file_id):
    """Get processing status for a file"""
    data = load_cache(file_id)
    return {
        "has_original": "original_text" in data,
        "has_corrected": "corrected_text" in data,
        "has_metadata": "metadata" in data,
        "has_vectors": "vectors" in data
    }
