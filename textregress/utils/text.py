"""
Text processing utilities.

This module contains functions for processing and manipulating text data.
"""

from typing import List, Optional
import numpy as np


def chunk_text(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """
    Split text into overlapping chunks of maximum length.

    Args:
        text: Input text to split
        max_length: Maximum length of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + max_length, text_length)
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end
        
    return chunks


def pad_chunks(chunks: List[str], max_length: int, pad_token: str = " ") -> List[str]:
    """
    Pad text chunks to a maximum length.

    Args:
        chunks: List of text chunks
        max_length: Maximum length to pad to
        pad_token: Token to use for padding

    Returns:
        List of padded text chunks
    """
    return [chunk + pad_token * (max_length - len(chunk)) for chunk in chunks] 