# common/utils.py
"""
Shared utilities for the enhanced agent pipeline.
Centralized to prevent code duplication and ensure consistency.
"""

import json
import re
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import sqlite3
from contextlib import contextmanager

# JSON Parsing Utilities
def extract_first_json_block(text: str) -> str:
    """Extract and clean JSON from potentially malformed text"""
    # Remove fenced code blocks if present
    text = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.DOTALL | re.MULTILINE)

    # Replace smart quotes/dashes with ASCII using regex
    text = re.sub(r'[""‟]', '"', text)  # Smart double quotes
    text = re.sub(r'[''‛]', "'", text)  # Smart single quotes
    text = re.sub(r'[–—‑]', '-', text)  # Various dashes

    # Find JSON boundaries
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in text")

    # Find matching closing brace
    brace_count = 0
    end_idx = -1
    for i, char in enumerate(text[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        raise ValueError("Unclosed JSON object")

    return text[start_idx:end_idx]

def repair_and_parse_json(text: str) -> Dict[str, Any]:
    """Robust JSON parsing with multiple repair strategies"""
    strategies = [
        # Strategy 1: Parse as-is
        lambda t: json.loads(t),

        # Strategy 2: Extract first JSON block
        lambda t: json.loads(extract_first_json_block(t)),

        # Strategy 3: Fix common issues and retry
        lambda t: json.loads(_fix_common_json_issues(t)),

        # Strategy 4: Remove problematic trailing content
        lambda t: json.loads(_clean_trailing_content(t)),
    ]

    for i, strategy in enumerate(strategies, 1):
        try:
            result = strategy(text)
            if i > 1:
                logging.warning(f"JSON parsed successfully using strategy {i}")
            return result
        except Exception as e:
            if i == len(strategies):
                logging.error(f"All JSON parsing strategies failed. Last error: {e}")
                raise ValueError(f"Unable to parse JSON after {len(strategies)} attempts: {e}")
            continue

def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Fix unquoted keys (simple cases)
    text = re.sub(r'(\w+):', r'"\1":', text)

    # Fix single quotes to double quotes (but preserve escaped quotes)
    text = re.sub(r"(?<!\\)'", '"', text)

    return text

def _clean_trailing_content(text: str) -> str:
    """Remove non-JSON content after valid JSON"""
    try:
        json_start = text.find('{')
        if json_start == -1:
            return text

        # Parse incrementally to find valid JSON boundary
        for end in range(len(text), json_start, -1):
            try:
                candidate = text[json_start:end]
                json.loads(candidate)
                return candidate
            except:
                continue
    except:
        pass
    return text

# Cache Management Utilities
class PersistentLLMCache:
    """SQLite-backed LLM cache with TTL and tagging support"""

    def __init__(self, cache_file: Path, default_ttl: int = 7200):
        self.cache_file = cache_file
        self.default_ttl = default_ttl
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    tags TEXT DEFAULT ''
                )
            """)
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)')

    def get(self, prompt: str, model: str = "default", **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        key = self._create_cache_key(prompt, model, **kwargs)
        current_time = time.time()

        with sqlite3.connect(self.cache_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT value, timestamp, ttl_seconds FROM cache_entries WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()

            if row:
                if (current_time - row['timestamp']) <= row['ttl_seconds']:
                    # Update hit count
                    conn.execute(
                        'UPDATE cache_entries SET hit_count = hit_count + 1 WHERE key = ?',
                        (key,)
                    )
                    try:
                        return json.loads(row['value'])
                    except json.JSONDecodeError:
                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                else:
                    # Expired entry, remove it
                    conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))

        return None

    def set(self, prompt: str, value: Dict[str, Any], model: str = "default", 
            ttl: Optional[int] = None, tags: List[str] = None, **kwargs):
        """Store response in cache"""
        key = self._create_cache_key(prompt, model, **kwargs)
        ttl = ttl or self.default_ttl
        tags_str = ','.join(tags or [])

        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, value, timestamp, ttl_seconds, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (key, json.dumps(value), time.time(), ttl, tags_str))

    def _create_cache_key(self, prompt: str, model: str = "default", **kwargs) -> str:
        """Create deterministic cache key"""
        content_hash = hashlib.sha256()
        content_hash.update(prompt.encode('utf-8'))
        content_hash.update(model.encode('utf-8'))

        for key in sorted(kwargs.keys()):
            content_hash.update(f"{key}:{kwargs[key]}".encode('utf-8'))

        return content_hash.hexdigest()[:16]

# Validation Utilities
def validate_with_detailed_errors(model_class, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data with detailed error reporting"""
    try:
        validated = model_class.model_validate(data)
        return {
            'success': True,
            'data': validated,
            'errors': [],
            'warnings': []
        }
    except Exception as e:
        errors = []
        warnings = []

        if hasattr(e, 'errors'):
            for error in e.errors():
                field_path = '.'.join(str(loc) for loc in error.get('loc', []))
                error_info = {
                    'field': field_path,
                    'message': error.get('msg', 'Unknown error'),
                    'type': error.get('type', 'unknown'),
                    'input': str(error.get('input', 'N/A'))
                }

                if error.get('type') in ['missing', 'extra_forbidden']:
                    errors.append(error_info)
                else:
                    warnings.append(error_info)

        return {
            'success': False,
            'data': None,
            'errors': errors,
            'warnings': warnings,
            'raw_error': str(e)
        }

# File System Utilities  
def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return path"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_file_write(file_path: Path, content: str, backup: bool = True) -> bool:
    """Safely write file with optional backup"""
    try:
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f'{file_path.suffix}.bak')
            backup_path.write_text(file_path.read_text())

        file_path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        logging.error(f"Failed to write {file_path}: {e}")
        return False

# Performance Monitoring
class PerformanceMonitor:
    """Simple performance monitoring utility"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing and return elapsed time"""
        if name not in self.start_times:
            return 0.0

        elapsed = time.time() - self.start_times[name]
        self.metrics[name] = self.metrics.get(name, [])
        self.metrics[name].append(elapsed)
        del self.start_times[name]
        return elapsed

# Context manager for performance monitoring
@contextmanager
def monitor_performance(monitor: PerformanceMonitor, operation_name: str):
    """Context manager for automatic performance monitoring"""
    monitor.start_timer(operation_name)
    try:
        yield
    finally:
        monitor.end_timer(operation_name)
