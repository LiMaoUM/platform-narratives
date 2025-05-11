import networkx as nx
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Set, Callable, Any, Optional, Union

class NarrativeTreeAnalyzer:
    """
    Analyzes narrative trees in social media conversations by extracting anchor posts
    and their replies, then analyzing the discourse patterns.
    
    This class provides methods for extracting tree structures from conversation graphs,
    calculating tree statistics, and analyzing discourse patterns using external analyzers.
    It serves as the third component in the platform narratives analysis pipeline, after
    FastLexRank for content significance and SimilarityGraphBuilder for cross-platform connections.
    """
    
    def __init__(self, llm_analyzer=None):
        """
        Initialize the narrative tree analyzer.
        
        Args:
            llm_analyzer: Optional callable that uses an LLM to analyze narrative content
                         Should take a post or list of posts and return analysis results
        """
        self.llm_analyzer = llm_analyzer
    
    def extract_tree_statistics(self, graph, root_id):
        """
        Calculate statistics for a conversation tree.
        
        Args:
            graph: NetworkX DiGraph of post relationships
            root_id: ID of the root post
            
        Returns:
            Dictionary with tree statistics (depth, breadth, num_nodes)
        """
        # Create subgraph of the tree
        tree_nodes = self.get_tree_nodes(graph, root_id)
        tree = graph.subgraph(tree_nodes).copy()
        
        if tree.number_of_nodes() == 0:
            return {
                "depth": 0,
                "breadth": 0,
                "num_nodes": 1  # Count the root node
            }
        
        # Calculate depth (longest path from root to leaf)
        depth = nx.dag_longest_path_length(tree)
        
        # Compute levels from root
        levels = nx.single_source_shortest_path_length(tree, root_id)
        
        # Count how many nodes at each level
        level_counts = {}
        for level in levels.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Breadth is the max number of nodes at the same level
        breadth = max(level_counts.values())
        
        # Total number of nodes
        num_nodes = tree.number_of_nodes()
        
        return {
            "depth": depth,
            "breadth": breadth,
            "num_nodes": num_nodes
        }
    
    def get_descendants(self, graph, post_id):
        """
        Get all descendants (replies) of a post in the graph.
        
        Args:
            graph: NetworkX DiGraph of post relationships
            post_id: ID of the post to find descendants for
            
        Returns:
            Set of post IDs that are descendants of the given post
        """
        if post_id in graph:
            return nx.descendants(graph, post_id)
        return set()
    
    def get_tree_nodes(self, graph, post_id):
        """
        Get a post and all its descendants as a tree.
        
        Args:
            graph: NetworkX DiGraph of post relationships
            post_id: ID of the root post
            
        Returns:
            Set of post IDs including the root and all descendants
        """
        descendants = self.get_descendants(graph, post_id)
        return {post_id} | descendants
    
    def extract_posts_from_trees(self, graph, anchor_ids, id_to_post_map):
        """
        Extract all posts and replies from conversation trees rooted at anchor posts.
        
        Args:
            graph: NetworkX DiGraph of post relationships
            anchor_ids: List of post IDs to use as anchor/root posts
            id_to_post_map: Dictionary mapping post IDs to post content
            
        Returns:
            List of posts from the trees
        """
        posts_and_replies = []
        
        for post_id in anchor_ids:
            # Get all nodes in the tree
            tree_nodes = self.get_tree_nodes(graph, post_id)
            
            # Get the post content for each node
            posts = [id_to_post_map.get(node_id) for node_id in tree_nodes]
            
            # Add to the collection
            posts_and_replies.extend(posts)
        
        # Remove None values (posts that weren't found in the map)
        posts_and_replies = [post for post in posts_and_replies if post]
        
        return posts_and_replies
    
    def analyze_discourse(self, posts, analyzer_function=None, batch_size=8):
        """
        Analyze the discourse in a collection of posts using a provided analyzer function.
        
        Args:
            posts: List of posts to analyze
            analyzer_function: Function that takes a post and returns an analysis
                              If None, uses the llm_analyzer provided during initialization
            batch_size: Number of posts to analyze in each batch
            
        Returns:
            List of analysis results
        """
        if analyzer_function is None:
            if self.llm_analyzer is None:
                raise ValueError("No analyzer function provided and no llm_analyzer set during initialization")
            analyzer_function = self.llm_analyzer
            
        results = []
        
        for i in tqdm(range(0, len(posts), batch_size)):
            batch = posts[i:i+batch_size]
            batch_results = [analyzer_function(post) for post in batch]
            results.extend(batch_results)
        
        return results
        
    def analyze_tree_narratives(self, graph, root_ids, id_to_post_map, analyzer_function=None, batch_size=8):
        """
        Extract and analyze narrative trees in one operation.
        
        Args:
            graph: NetworkX DiGraph of post relationships
            root_ids: List of post IDs to use as anchor/root posts
            id_to_post_map: Dictionary mapping post IDs to post content
            analyzer_function: Function to analyze posts (uses llm_analyzer if None)
            batch_size: Number of posts to analyze in each batch
            
        Returns:
            Dictionary with tree statistics and analysis results
        """
        # Extract posts from trees
        posts = self.extract_posts_from_trees(graph, root_ids, id_to_post_map)
        
        # Calculate statistics for each tree
        tree_stats = {}
        for root_id in root_ids:
            tree_stats[root_id] = self.extract_tree_statistics(graph, root_id)
        
        # Analyze discourse in the posts
        analysis_results = self.analyze_discourse(posts, analyzer_function, batch_size)
        
        return {
            "posts": posts,
            "tree_stats": tree_stats,
            "analysis_results": analysis_results
        }
        
    def get_narrative_summary(self, analysis_results):
        """
        Generate a summary of narrative patterns from analysis results.
        
        Args:
            analysis_results: Results from analyze_discourse or analyze_tree_narratives
            
        Returns:
            Dictionary with narrative patterns and summary statistics
        """
        # This is a placeholder for more sophisticated analysis
        # In a real implementation, this would aggregate and summarize the LLM analysis results
        
        # Count occurrences of different narrative elements
        narrative_elements = {}
        for result in analysis_results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in narrative_elements:
                        narrative_elements[key] = []
                    narrative_elements[key].append(value)
        
        # Calculate summary statistics
        summary = {}
        for key, values in narrative_elements.items():
            if isinstance(values[0], (int, float)):
                summary[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            elif isinstance(values[0], str):
                # Count occurrences of each unique value
                value_counts = {}
                for value in values:
                    value_counts[value] = value_counts.get(value, 0) + 1
                summary[key] = value_counts
        
        return summary