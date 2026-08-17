"""
Performance optimization module for AI-Powered Smart OCR
Provides caching, threading, and memory optimization
"""

import os
import json
import hashlib
import time
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import threading

logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy types to regular Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


class PerformanceOptimizer:
    """
    Performance optimization for OCR processing
    """
    
    def __init__(self, cache_dir: str = "data/cache", enable_cache: bool = True):
        """
        Initialize performance optimizer
        
        Args:
            cache_dir: Directory for caching results
            enable_cache: Whether to enable result caching
        """
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
    def _get_cache_key(self, image_path: str, languages: tuple, options: dict) -> str:
        """
        Generate a unique cache key based on input parameters
        
        Args:
            image_path: Path to the image file
            languages: Tuple of language codes
            options: Dictionary of processing options
            
        Returns:
            Unique cache key
        """
        # Create a hash based on file content and parameters
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        params_hash = hashlib.md5(
            json.dumps({
                'languages': sorted(languages),
                'options': sorted(options.items())
            }, sort_keys=True).encode()
        ).hexdigest()
        
        return f"{file_hash}_{params_hash}"
    
    def get_cached_result(self, image_path: str, languages: tuple, options: dict) -> Optional[Dict]:
        """
        Get cached result if available
        
        Args:
            image_path: Path to the image file
            languages: Tuple of language codes
            options: Dictionary of processing options
            
        Returns:
            Cached result or None
        """
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(image_path, languages, options)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid (24 hours)
                cache_time = cached_data.get('timestamp', 0)
                if time.time() - cache_time < 86400:  # 24 hours
                    logger.info(f"Using cached result for {image_path}")
                    return cached_data.get('result')
                else:
                    # Remove expired cache
                    cache_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")
        
        return None
    
    def cache_result(self, image_path: str, languages: tuple, options: dict, result: Dict):
        """
        Cache a processing result
        
        Args:
            image_path: Path to the image file
            languages: Tuple of language codes
            options: Dictionary of processing options
            result: Processing result to cache
        """
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(image_path, languages, options)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with self.lock:
                cache_data = {
                    'timestamp': time.time(),
                    'result': convert_to_serializable(result)
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)
                
                logger.info(f"Cached result for {image_path}")
                
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def clear_cache(self, older_than_hours: int = 24):
        """
        Clear cache entries older than specified hours
        
        Args:
            older_than_hours: Remove cache entries older than this many hours
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)
            
            removed_count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if cache_data.get('timestamp', 0) < cutoff_time:
                        cache_file.unlink()
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process cache file {cache_file}: {e}")
            
            logger.info(f"Cleared {removed_count} cache entries older than {older_than_hours} hours")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            current_time = time.time()
            recent_count = 0
            old_count = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if current_time - cache_data.get('timestamp', 0) < 86400:  # 24 hours
                        recent_count += 1
                    else:
                        old_count += 1
                        
                except Exception:
                    pass
            
            return {
                'total_entries': len(cache_files),
                'recent_entries': recent_count,
                'old_entries': old_count,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_enabled': self.enable_cache
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                'error': str(e),
                'cache_enabled': self.enable_cache
            }


class MemoryOptimizer:
    """
    Memory optimization for large batch processing
    """
    
    def __init__(self, max_memory_mb: int = 4096):
        """
        Initialize memory optimizer
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        
    def estimate_image_memory(self, image_path: str) -> float:
        """
        Estimate memory usage for processing an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Estimated memory usage in MB
        """
        try:
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            # Estimate 3x file size for processing (image loading + preprocessing + OCR)
            estimated_memory = file_size_mb * 3
            return estimated_memory
        except Exception as e:
            logger.warning(f"Failed to estimate memory for {image_path}: {e}")
            return 50.0  # Conservative default
    
    def can_process_image(self, image_path: str) -> bool:
        """
        Check if an image can be processed given current memory constraints
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if processing is possible, False otherwise
        """
        estimated_memory = self.estimate_image_memory(image_path)
        return (self.current_memory_mb + estimated_memory) < self.max_memory_mb
    
    def allocate_memory(self, image_path: str):
        """
        Allocate memory for processing an image
        
        Args:
            image_path: Path to the image file
        """
        estimated_memory = self.estimate_image_memory(image_path)
        self.current_memory_mb += estimated_memory
    
    def free_memory(self, image_path: str):
        """
        Free memory after processing an image
        
        Args:
            image_path: Path to the image file
        """
        estimated_memory = self.estimate_image_memory(image_path)
        self.current_memory_mb = max(0, self.current_memory_mb - estimated_memory)
    
    def get_memory_status(self) -> Dict:
        """
        Get current memory status
        
        Returns:
            Dictionary with memory status information
        """
        return {
            'current_memory_mb': round(self.current_memory_mb, 2),
            'max_memory_mb': self.max_memory_mb,
            'available_memory_mb': round(self.max_memory_mb - self.current_memory_mb, 2),
            'usage_percentage': round((self.current_memory_mb / self.max_memory_mb) * 100, 2)
        }


def main():
    """Example usage of performance optimizer"""
    optimizer = PerformanceOptimizer(enable_cache=True)
    
    # Get cache statistics
    stats = optimizer.get_cache_stats()
    print("Cache Statistics:", json.dumps(stats, indent=2))
    
    # Clear old cache entries
    optimizer.clear_cache(older_than_hours=48)
    
    # Memory optimizer example
    memory_optimizer = MemoryOptimizer(max_memory_mb=2048)
    print("Memory Status:", json.dumps(memory_optimizer.get_memory_status(), indent=2))


if __name__ == "__main__":
    main()