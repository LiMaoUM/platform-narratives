#!/usr/bin/env python3
"""
Example script demonstrating the complete narrative analysis workflow.

This script shows how to use the three main components of the platform narratives analysis:
1. FastLexRank for identifying significant content
2. SimilarityGraphBuilder for cross-platform connections
3. NarrativeTreeAnalyzer for discourse analysis
"""

import pandas as pd
import networkx as nx
import json
from tqdm.auto import tqdm

# Import the project components
from src.lexrank import FastLexRank
from src.similarity_graph import SimilarityGraphBuilder
from src.narrative_trees import NarrativeTreeAnalyzer
from src.utils import load_json_data, create_id_to_post_map
from src.graph_analysis import build_graph

# Optional: Define a simple LLM analyzer function (replace with actual LLM integration)
def simple_llm_analyzer(post):
    """
    A placeholder for LLM-based narrative analysis.
    In a real implementation, this would call an LLM API.
    
    Args:
        post: Post content to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # This is a placeholder - replace with actual LLM call
    text = post.get('post', '') if isinstance(post, dict) else str(post)
    
    # Simple keyword-based analysis as placeholder
    analysis = {
        'sentiment': 'neutral',
        'topics': [],
        'narrative_elements': []
    }
    
    # Very basic keyword detection (replace with actual LLM analysis)
    negative_words = ['bad', 'terrible', 'awful', 'wrong', 'hate']
    positive_words = ['good', 'great', 'excellent', 'right', 'love']
    
    text_lower = text.lower()
    
    # Simple sentiment analysis
    neg_count = sum(1 for word in negative_words if word in text_lower)
    pos_count = sum(1 for word in positive_words if word in text_lower)
    
    if pos_count > neg_count:
        analysis['sentiment'] = 'positive'
    elif neg_count > pos_count:
        analysis['sentiment'] = 'negative'
    
    return analysis

def main():
    # Step 1: Load data (replace with your actual data paths)
    try:
        # Example data loading - adjust paths as needed
        print("Loading data...")
        platform1_data = load_json_data('data/platform1_posts.json')
        platform2_data = load_json_data('data/platform2_posts.json')
        platform3_data = load_json_data('data/platform3_posts.json')
    except FileNotFoundError:
        print("Example data files not found. This is just a demonstration script.")
        print("In a real scenario, you would load your actual data files.")
        # Create some dummy data for demonstration
        platform1_data = [{"id": f"p1_{i}", "post": f"Platform 1 post {i}"} for i in range(10)]
        platform2_data = [{"id": f"p2_{i}", "post": f"Platform 2 post {i}"} for i in range(10)]
        platform3_data = [{"id": f"p3_{i}", "post": f"Platform 3 post {i}"} for i in range(10)]
    
    # Step 2: Identify significant content using FastLexRank
    print("\nIdentifying significant content with FastLexRank...")
    lexrank = FastLexRank()
    
    # Convert to DataFrames for FastLexRank
    df1 = pd.DataFrame(platform1_data)
    df2 = pd.DataFrame(platform2_data)
    df3 = pd.DataFrame(platform3_data)
    
    # Rank posts by significance
    ranked_df1 = lexrank.rank(df1)
    ranked_df2 = lexrank.rank(df2)
    ranked_df3 = lexrank.rank(df3)
    
    print(f"Top 3 significant posts from platform 1:")
    for i, row in ranked_df1.head(3).iterrows():
        print(f"  Score: {row['ap']:.4f} | {row['post'][:50]}...")
    
    # Step 3: Build cross-platform similarity graph
    print("\nBuilding cross-platform similarity graph...")
    graph_builder = SimilarityGraphBuilder(similarity_threshold=0.6)
    
    # Prepare data for tripartite graph
    platform_data = {
        'platform1': ranked_df1['post'].tolist(),
        'platform2': ranked_df2['post'].tolist(),
        'platform3': ranked_df3['post'].tolist()
    }
    
    # Build the graph (in a real scenario, this might take time)
    print("This would build a tripartite graph connecting similar posts across platforms.")
    print("Skipping actual computation for this example.")
    
    # Step 4: Analyze narrative trees
    print("\nAnalyzing narrative trees...")
    
    # Create a sample conversation graph (in a real scenario, this would come from your data)
    # For demonstration, we'll create a simple tree structure
    conversation_data = [
        {"id": "root1", "parent_id": None, "post": "This is the first root post"},
        {"id": "reply1", "parent_id": "root1", "post": "This is a reply to the first post"},
        {"id": "reply2", "parent_id": "root1", "post": "This is another reply to the first post"},
        {"id": "reply3", "parent_id": "reply1", "post": "This is a reply to the first reply"},
        {"id": "root2", "parent_id": None, "post": "This is the second root post"},
        {"id": "reply4", "parent_id": "root2", "post": "This is a reply to the second post"}
    ]
    
    # Build the conversation graph
    graph = build_graph(conversation_data)
    
    # Create ID to post mapping
    id_to_post_map = create_id_to_post_map(conversation_data)
    
    # Initialize the narrative tree analyzer with our LLM analyzer
    tree_analyzer = NarrativeTreeAnalyzer(llm_analyzer=simple_llm_analyzer)
    
    # Analyze the narrative trees
    root_ids = ["root1", "root2"]
    analysis_results = tree_analyzer.analyze_tree_narratives(
        graph, root_ids, id_to_post_map
    )
    
    # Print tree statistics
    print("\nTree Statistics:")
    for root_id, stats in analysis_results["tree_stats"].items():
        print(f"  Tree {root_id}: Depth={stats['depth']}, Breadth={stats['breadth']}, Nodes={stats['num_nodes']}")
    
    # Generate narrative summary
    print("\nNarrative Summary:")
    summary = tree_analyzer.get_narrative_summary(analysis_results["analysis_results"])
    print(json.dumps(summary, indent=2))
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()