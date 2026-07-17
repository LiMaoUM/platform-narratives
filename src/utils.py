"""Utility functions for the platform narratives analysis.

This module provides helper functions for data loading, processing,
and other common operations.
"""

import json
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Dict, Any

def load_json_data(file_path):
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(file=file_path) as f:
        return json.load(f)

def create_id_to_post_map(posts, id_key='id'):
    """Create a mapping from post IDs to post objects.
    
    Args:
        posts: List of post objects
        id_key: Key in the post object that contains the ID
        
    Returns:
        Dictionary mapping post IDs to post objects
    """
    return {post.get(id_key): post for post in posts if post.get(id_key)}

def posts_to_dataframe(posts, text_key='post'):
    """Convert a list of posts to a pandas DataFrame.
    
    Args:
        posts: List of post objects or dictionaries
        text_key: Key in the post object that contains the text content
        
    Returns:
        DataFrame containing the posts
    """
    if isinstance(posts[0], dict):
        return pd.DataFrame(posts)
    else:
        return pd.DataFrame({text_key: posts})

def extract_anchor_ids(stats_df, match_column='matched_id', id_column='post_id'):
    """Extract anchor post IDs from a stats DataFrame.
    
    Args:
        stats_df: DataFrame containing post statistics
        match_column: Column name for the match ID
        id_column: Column name for the post ID
        
    Returns:
        Array of anchor post IDs
    """
    return stats_df.loc[stats_df[match_column] == 0, id_column].values

def setup_logging(level: str = "INFO"):
    """Setup logging configuration for the analysis pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('analysis.log')
        ]
    )

def save_pipeline_state(state: Dict[str, Any], filepath: Path):
    """Save pipeline state to JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving pipeline state: {e}")

def load_pipeline_state(filepath: Path) -> Dict[str, Any]:
    """Load pipeline state from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading pipeline state: {e}")
        return {}