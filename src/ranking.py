"""Ranking module for identifying significant content.

This module implements the FastLexRank algorithm for identifying
significant content in a collection of posts.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def fastLexRank(df):
    """Identify significant content using FastLexRank algorithm.
    
    Args:
        df: DataFrame containing a 'post' column with text content
        
    Returns:
        DataFrame with an additional 'ap' column containing significance scores
    """
    posts = df["post"]
    # Remove @mentions
    posts = posts.str.replace(r'@\w+', '', regex=True)
    # Remove hashtags
    posts = posts.str.replace(r'#\w+', '', regex=True)
    # Remove URLs
    posts = posts.str.replace(r'http\S+', '', regex=True)
    
    # Generate embeddings for all posts
    embeddings = model.encode(posts, show_progress_bar=True)
    
    # Sum embeddings in column
    z = embeddings.sum(axis=0)
    
    # Normalize the sum
    z = z / np.sqrt((z**2).sum(axis=0))
    
    # Calculate alignment scores
    ap = np.dot(embeddings, z)
    
    # Add scores to dataframe
    df["ap"] = ap
    
    # Sort by alignment score
    return df.sort_values(by="ap", ascending=False)