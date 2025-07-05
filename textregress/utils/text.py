"""
Text processing utilities.

This module contains functions for processing and manipulating text data.
"""

from typing import List, Optional
import numpy as np


def chunk_text(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """
    Splits the text into chunks.
    
    If max_length is provided, the text is split into overlapping chunks.
    Only full chunks (i.e. with exactly max_length words) are returned.
    If the text is shorter than max_length, the entire text is returned as a single chunk.
    Otherwise, the entire text is returned as a single chunk.
    
    Args:
        text (str): The input text.
        max_length (int): Maximum number of words per chunk.
        overlap (int): Number of words to overlap between chunks.
        
    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
    
    if max_length:
        words = text.split()
        
        # If text is shorter than max_length, return the entire text
        if len(words) <= max_length:
            return [text]
        
        chunks = []
        i = 0
        
        # Safety check to prevent infinite loop
        if overlap >= max_length:
            overlap = max_length - 1
        
        while i + max_length <= len(words):
            chunk = " ".join(words[i:i+max_length])
            chunks.append(chunk)
            i += max(max_length - overlap, 1)
            
            # Additional safety check to prevent infinite loop
            if i >= len(words):
                break
                
        return chunks
    else:
        return [text]


def pad_chunks(chunks: List[str], padding_value: int = 0) -> List[str]:
    """
    Placeholder for padding chunks if necessary.
    Currently, this returns the chunks as-is.
    
    Args:
        chunks (list of str): List of text chunks.
        padding_value (int): Padding value.
        
    Returns:
        List[str]: Padded chunks.
    """
    return chunks 