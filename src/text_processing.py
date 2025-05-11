"""Text processing module for cleaning and preparing text data.

This module provides functions for cleaning and preprocessing text data
from social media posts.
"""

import re
from bs4 import BeautifulSoup
from fast_langdetect import detect_language

def clean_text(text):
    """Clean text by removing HTML, mentions, hashtags, and URLs.
    
    Args:
        text: String text to clean
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
        
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    return text.strip()

def detect_post_language(text):
    """Detect the language of a post.
    
    Args:
        text: String text to detect language for
        
    Returns:
        Language code string
    """
    try:
        return detect_language(text)
    except Exception:
        return "unknown"

def filter_posts_by_language(posts, language="en"):
    """Filter posts by detected language.
    
    Args:
        posts: List of post texts or dictionary with 'post' key
        language: Language code to filter for (default: 'en' for English)
        
    Returns:
        List of posts in the specified language
    """
    filtered_posts = []
    
    for post in posts:
        if isinstance(post, dict):
            text = post.get('post', '')
        else:
            text = post
            
        if detect_post_language(text) == language:
            filtered_posts.append(post)
            
    return filtered_posts