#!/usr/bin/env python3
"""
Example script demonstrating how to use the platform narratives analysis tools.

This script shows a complete workflow for analyzing social media posts,
identifying significant content, and extracting post relationships.
"""

import os
import pandas as pd

# Import components from our package
from src.ranking import fastLexRank
from src.graph_analysis import build_graph, get_posts_from_tree
from src.text_processing import clean_text, filter_posts_by_language
from src.utils import load_json_data, create_id_to_post_map, extract_anchor_ids

def main():
    # Step 1: Load data
    # Replace with your data path
    data_path = os.path.join('data', 'your_data.json')
    
    # For demonstration, we'll handle the case where the file doesn't exist yet
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please add your data file.")
        print("Expected format: JSON file with posts containing 'id', 'parent_id', and 'post' fields.")
        return
    
    # Load the data
    data = load_json_data(data_path)
    print(f"Loaded {len(data)} posts from {data_path}")
    
    # Step 2: Clean and preprocess the data
    # Clean post text
    for post in data:
        if 'post' in post:
            post['post'] = clean_text(post['post'])
    
    # Filter to English posts only
    english_posts = filter_posts_by_language(data)
    print(f"Filtered to {len(english_posts)} English posts")
    
    # Step 3: Build the post graph
    post_graph = build_graph(english_posts)
    print(f"Built graph with {post_graph.number_of_nodes()} nodes and {post_graph.number_of_edges()} edges")
    
    # Step 4: Create a mapping from post IDs to posts
    id_to_post = create_id_to_post_map(english_posts)
    
    # Step 5: Convert posts to DataFrame for ranking
    posts_df = pd.DataFrame(english_posts)
    
    # Step 6: Apply FastLexRank to identify significant content
    ranked_df = fastLexRank(posts_df)
    print("Applied FastLexRank to identify significant content")
    
    # Step 7: Extract anchor posts (posts with no parent or matched_id=0)
    # For demonstration, we'll use the top 10 ranked posts as anchors
    top_posts = ranked_df.head(10)
    anchor_ids = top_posts['id'].values
    print(f"Selected {len(anchor_ids)} anchor posts")
    
    # Step 8: Extract all posts in the trees rooted at anchor posts
    tree_posts = get_posts_from_tree(post_graph, anchor_ids, id_to_post)
    print(f"Extracted {len(tree_posts)} posts from anchor post trees")
    
    # Step 9: Analyze the results
    # For example, print the top 5 most significant posts
    print("\nTop 5 most significant posts:")
    for i, (_, row) in enumerate(ranked_df.head(5).iterrows(), 1):
        print(f"{i}. Score: {row['ap']:.4f} - {row['post'][:100]}...")

if __name__ == "__main__":
    main()