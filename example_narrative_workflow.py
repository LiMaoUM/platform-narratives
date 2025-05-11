#!/usr/bin/env python3
"""
Example workflow demonstrating the complete platform narratives analysis pipeline.

This script shows how to use all the components of the platform narratives analysis:
1. FastLexRank for identifying significant content
2. SimilarityGraphBuilder for cross-platform connections
3. NarrativeClassifier for LLM-based narrative analysis
4. ReplyChainAnalyzer for discourse dynamics analysis
5. NarrativeTreeAnalyzer for conversation structure analysis
"""

import pandas as pd
import networkx as nx
import json
from tqdm.auto import tqdm

# Import the project components
from src.lexrank import FastLexRank
from src.similarity_graph import SimilarityGraphBuilder
from src.narrative_classification import NarrativeClassifier
from src.reply_chain_analysis import ReplyChainAnalyzer
from src.narrative_trees import NarrativeTreeAnalyzer
from src.utils import load_json_data, create_id_to_post_map
from src.graph_analysis import build_graph

# Define a simple LLM pipeline function (replace with actual implementation)
def simple_llm_pipeline(messages, max_new_tokens=200):
    """
    A placeholder for LLM pipeline.
    In a real implementation, this would call an LLM API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        List with a single dictionary containing generated text
    """
    # This is a placeholder - replace with actual LLM call
    prompt = messages[-1]['content']
    
    # Very basic response generation (replace with actual LLM)
    response = "This is a placeholder response. Replace with actual LLM integration."
    
    if "classify" in prompt.lower():
        response = '{"narrative_frame": "media bias", "main_subject": "Trump", "stance": "supportive", "topic_focus": "media"}'
    elif "reinforce, challenge, or shift" in prompt.lower():
        response = "reinforce"
    
    return [{
        "generated_text": [
            {"role": "assistant", "content": response}
        ]
    }]

def main():
    # Step 1: Load data (replace with your actual data paths)
    try:
        # Example data loading - adjust paths as needed
        print("Loading data...")
        platform1_data = load_json_data('data/platform1_posts.json')
        platform2_data = load_json_data('data/platform2_posts.json')
        platform3_data = load_json_data('data/platform3_posts.json')
    except FileNotFoundError:
        print("Example data files not found. Creating dummy data for demonstration.")
        # Create some dummy data for demonstration
        platform1_data = [{
            "id": f"p1_{i}", 
            "post": f"Platform 1 post {i} about media bias and fake news."
        } for i in range(10)]
        
        platform2_data = [{
            "id": f"p2_{i}", 
            "post": f"Platform 2 post {i} discussing Washington Post reporting."
        } for i in range(10)]
        
        platform3_data = [{
            "id": f"p3_{i}", 
            "post": f"Platform 3 post {i} with political commentary."
        } for i in range(10)]
    
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
        print(f"  Score: {row.get('ap', 0):.4f} | {row.get('post', '')[:50]}...")
    
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
    
    # Step 4: Perform narrative classification
    print("\nPerforming narrative classification with LLM...")
    
    # Initialize the narrative classifier with our LLM pipeline
    classifier = NarrativeClassifier(llm_pipeline=simple_llm_pipeline)
    
    # Example component posts for classification
    component_posts = {
        'platform1': ["The Fake News Washington Post is at it again with their lies."],
        'platform2': ["Washington Post reporting is factual and well-researched."],
        'platform3': ["Media bias exists across the spectrum in different forms."]
    }
    
    # Classify the component posts
    classification_results = classifier.classify_component_posts(component_posts)
    
    print("\nNarrative Classification Results:")
    for result in classification_results:
        platform = result.get('platform', '')
        classification = result.get('classification', {})
        print(f"  Platform: {platform}")
        print(f"  Classification: {json.dumps(classification, indent=2)}")
    
    # Step 5: Analyze reply chains
    print("\nAnalyzing reply chains...")
    
    # Initialize the reply chain analyzer with our LLM pipeline
    reply_analyzer = ReplyChainAnalyzer(llm_pipeline=simple_llm_pipeline)
    
    # Example root post and replies
    root_post = "The Fake News Washington Post came up with the ridiculous idea that Donald J. Trump will call for Mandatory Military Service. This is only a continuation of their EIGHT YEAR failed attempt to damage me with the Voters. The Story is completely untrue. In fact, I never even thought of that idea. Only a degenerate former Newspaper, which has lost 50% of its Readers, would fabricate such a tale. Just another Fake Story, one of many, made up by the DEAD Washington Compost!"
    
    replies = [
        "Absolutely right! The Washington Post is nothing but fake news!",
        "I trust the Washington Post's reporting. They have credible sources.",
        "I'm more concerned about the economy than this story."
    ]
    
    # Analyze the reply chain
    reply_analysis = reply_analyzer.analyze_reply_chain(root_post, replies)
    
    print("\nReply Chain Analysis Results:")
    print(f"  Category: {reply_analysis.get('category', '')}")
    
    # Step 6: Analyze narrative trees
    print("\nAnalyzing narrative trees...")
    
    # Create a sample conversation graph (in a real scenario, this would come from your data)
    conversation_data = [
        {"id": "root1", "parent_id": None, "post": root_post},
        {"id": "reply1", "parent_id": "root1", "post": replies[0]},
        {"id": "reply2", "parent_id": "root1", "post": replies[1]},
        {"id": "reply3", "parent_id": "reply1", "post": "This is a reply to the first reply"}
    ]
    
    # Build the conversation graph
    graph = build_graph(conversation_data)
    
    # Create ID to post mapping
    id_to_post_map = create_id_to_post_map(conversation_data)
    
    # Initialize the narrative tree analyzer with our LLM analyzer
    tree_analyzer = NarrativeTreeAnalyzer(llm_analyzer=lambda post: {"sentiment": "neutral"})
    
    # Analyze the narrative trees
    root_ids = ["root1"]
    analysis_results = tree_analyzer.analyze_tree_narratives(
        graph, root_ids, id_to_post_map
    )
    
    # Print tree statistics
    print("\nTree Statistics:")
    for root_id, stats in analysis_results.get("tree_stats", {}).items():
        print(f"  Tree {root_id}: Depth={stats.get('depth')}, Breadth={stats.get('breadth')}, Nodes={stats.get('num_nodes')}")
    
    print("\nAnalysis complete!")
    print("\nTo use this with your own data:")
    print("1. Replace the placeholder LLM pipeline with your actual LLM integration")
    print("2. Load your real data instead of the dummy examples")
    print("3. Adjust parameters like similarity thresholds and batch sizes as needed")

if __name__ == "__main__":
    main()