"""
Simple In-Memory Cache Utility

Lightweight caching for API responses without external dependencies.

Usage:
    from src.core.utils import SimpleCache
    cache = SimpleCache(duration=300)  # 5 minutes
    
    # Store
    cache.set('player_123', player_data)
    
    # Retrieve
    data = cache.get('player_123')  # Returns data if <5 mins old, else None
"""

import time

class SimpleCache:
    """
    Time-based in-memory cache.
    
    Attributes:
        cache (dict): Storage {key: (timestamp, data)}
        duration (int): Seconds until expiration
        
    Methods:
        get(key): Retrieve if not expired
        set(key, data): Store with current timestamp
        
    Note:
        Cache is NOT persistent (resets when program ends)
        Not thread-safe (fine for single-threaded use)
    """

    def __init__(self, duration=300):
        self.cache = {}
        self.duration = duration  # 5 minutes default

    def get(self, key):
        current_time = time.time()
        if key in self.cache:
            timestamp, data = self.cache[key]
            if current_time - timestamp < self.duration:
                return data
        return None

    def set(self, key, data):
        self.cache[key] = (time.time(), data)
