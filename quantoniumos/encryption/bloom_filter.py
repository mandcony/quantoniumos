"""
QuantoniumOS - Bloom Filter Implementation

This module implements a space-efficient probabilistic data structure
for testing whether an element is a member of a set. False positives
are possible, but false negatives are not.

Used by the quantum-symbolic hash verification system.
"""

import hashlib
import math
from typing import Any, List, Optional, Tuple

from encryption.geometric_container import GeometricContainer


class BloomFilter:
    """
    Bloom filter implementation for efficient hash verification.
    
    Attributes:
        size: Size of the bit array
        hash_functions: Number of hash functions to use
        bit_array: The bit array to store values
        items_count: Number of items added to the filter
    """
    
    def __init__(self, size: int, hash_functions: int) -> None:
        """
        Initialize a new Bloom filter.
        
        Args:
            size: Size of the bit array
            hash_functions: Number of hash functions to use
        """
        self.size = size
        self.hash_functions = hash_functions
        self.bit_array = [False] * size
        self.items_count = 0
        
        # Validate parameters
        if size <= 0:
            raise ValueError("Size must be positive")
        if hash_functions <= 0:
            raise ValueError("Number of hash functions must be positive")
    
    def add(self, item: Any) -> None:
        """
        Add an item to the filter.
        
        Args:
            item: Item to add
        """
        for i in range(self.hash_functions):
            index = self._get_hash_index(item, i)
            self.bit_array[index] = True
        
        self.items_count += 1
    
    def add_batch(self, items: List[Any]) -> None:
        """
        Add multiple items to the filter.
        
        Args:
            items: List of items to add
        """
        for item in items:
            self.add(item)
    
    def contains(self, item: Any) -> bool:
        """
        Check if the filter might contain an item.
        
        Args:
            item: Item to check
            
        Returns:
            True if the item might be in the filter, False if definitely not
        """
        for i in range(self.hash_functions):
            index = self._get_hash_index(item, i)
            if not self.bit_array[index]:
                return False
        return True
    
    def get_indices(self, item: Any) -> List[int]:
        """
        Get the bit indices for an item.
        
        Args:
            item: Item to check
            
        Returns:
            List of bit indices
        """
        indices = []
        for i in range(self.hash_functions):
            index = self._get_hash_index(item, i)
            indices.append(index)
        return indices
    
    def _get_hash_index(self, item: Any, hash_function_index: int) -> int:
        """
        Get the hash index for an item using a specific hash function.
        
        Args:
            item: Item to hash
            hash_function_index: Index of the hash function to use
            
        Returns:
            Hash index
        """
        # Convert item to bytes if it's not already
        if not isinstance(item, bytes):
            item_bytes = str(item).encode('utf-8')
        else:
            item_bytes = item
        
        # Use different hash functions based on the index
        if hash_function_index == 0:
            # Use SHA-256
            h = hashlib.sha256(item_bytes).digest()
        elif hash_function_index == 1:
            # Use SHA-1
            h = hashlib.sha1(item_bytes).digest()
        elif hash_function_index == 2:
            # Use MD5
            h = hashlib.md5(item_bytes).digest()
        else:
            # For additional hash functions, combine earlier ones
            seed = hash_function_index.to_bytes(4, byteorder='big')
            h = hashlib.sha256(item_bytes + seed).digest()
        
        # Convert to an index
        return int.from_bytes(h[:4], byteorder='big') % self.size
    
    @staticmethod
    def optimal_size(expected_items: int, error_rate: float) -> Tuple[int, int]:
        """
        Calculate the optimal size and number of hash functions.
        
        Args:
            expected_items: Expected number of items
            error_rate: Desired false positive rate
            
        Returns:
            Tuple of (optimal_size, optimal_hash_functions)
        """
        size = -int(expected_items * math.log(error_rate) / (math.log(2) ** 2))
        hash_functions = int(size / expected_items * math.log(2))
        
        return max(size, 1), max(hash_functions, 1)
    
    def get_geometric_container(self) -> GeometricContainer:
        """
        Get a geometric container representation of the filter.
        
        Returns:
            Geometric container with the filter state
        """
        # Create a container with the filter state
        container = GeometricContainer(self.size)
        
        # Add active bits as points
        for i, bit in enumerate(self.bit_array):
            if bit:
                # Map bit index to a position in the container
                x = i % int(math.sqrt(self.size))
                y = i // int(math.sqrt(self.size))
                container.add_point(x, y)
        
        return container
    
    def union(self, other: 'BloomFilter') -> Optional['BloomFilter']:
        """
        Create a union of two bloom filters.
        
        Args:
            other: Another bloom filter
            
        Returns:
            New bloom filter with the union
        """
        if self.size != other.size or self.hash_functions != other.hash_functions:
            return None
        
        result = BloomFilter(self.size, self.hash_functions)
        result.bit_array = [a or b for a, b in zip(self.bit_array, other.bit_array)]
        result.items_count = self.items_count + other.items_count
        
        return result
    
    def intersection(self, other: 'BloomFilter') -> Optional['BloomFilter']:
        """
        Create an intersection of two bloom filters.
        
        Args:
            other: Another bloom filter
            
        Returns:
            New bloom filter with the intersection
        """
        if self.size != other.size or self.hash_functions != other.hash_functions:
            return None
        
        result = BloomFilter(self.size, self.hash_functions)
        result.bit_array = [a and b for a, b in zip(self.bit_array, other.bit_array)]
        result.items_count = min(self.items_count, other.items_count)
        
        return result
    
    def clear(self) -> None:
        """Reset the bloom filter."""
        self.bit_array = [False] * self.size
        self.items_count = 0
    
    def get_stats(self) -> dict:
        """
        Get statistics about the bloom filter.
        
        Returns:
            Dictionary of statistics
        """
        active_bits = sum(1 for bit in self.bit_array if bit)
        fill_ratio = active_bits / self.size
        
        # Calculate estimated false positive rate
        if self.items_count > 0:
            estimated_error = (1 - math.exp(-self.hash_functions * self.items_count / self.size)) ** self.hash_functions
        else:
            estimated_error = 0.0
        
        return {
            "size": self.size,
            "hash_functions": self.hash_functions,
            "items_count": self.items_count,
            "active_bits": active_bits,
            "fill_ratio": fill_ratio,
            "estimated_error": estimated_error
        }