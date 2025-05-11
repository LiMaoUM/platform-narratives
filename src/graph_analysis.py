"""Graph analysis module for processing post relationships.

This module provides functions for building and analyzing graph structures
of posts and their relationships.
"""

import networkx as nx

def build_graph(posts_data, parent_key='parent_id', child_key='id'):
    """Build a directed graph from post relationships.
    
    Args:
        posts_data: List of post dictionaries containing parent-child relationships
        parent_key: Key in the post dictionary for the parent ID
        child_key: Key in the post dictionary for the child/post ID
        
    Returns:
        NetworkX DiGraph representing the post relationships
    """
    graph = nx.DiGraph()
    
    for post in posts_data:
        child = post.get(child_key)
        parent = post.get(parent_key)
        
        if parent and child and parent != child:
            graph.add_edge(parent, child)
        elif child:
            graph.add_node(child)
            
    return graph

def get_descendants(graph, post_id):
    """Get all descendants (replies) of a post in the graph.
    
    Args:
        graph: NetworkX DiGraph of post relationships
        post_id: ID of the post to find descendants for
        
    Returns:
        Set of post IDs that are descendants of the given post
    """
    if post_id in graph:
        return nx.descendants(graph, post_id)
    return set()

def get_tree_nodes(graph, post_id):
    """Get a post and all its descendants as a tree.
    
    Args:
        graph: NetworkX DiGraph of post relationships
        post_id: ID of the root post
        
    Returns:
        Set of post IDs including the root and all descendants
    """
    descendants = get_descendants(graph, post_id)
    return {post_id} | descendants

def get_posts_from_tree(graph, root_ids, id_to_post_map):
    """Extract posts from a tree structure based on root IDs.
    
    Args:
        graph: NetworkX DiGraph of post relationships
        root_ids: List of root post IDs to extract trees for
        id_to_post_map: Dictionary mapping post IDs to post objects
        
    Returns:
        List of post objects from the trees
    """
    all_posts = []
    
    for post_id in root_ids:
        tree_nodes = get_tree_nodes(graph, post_id)
        posts = [id_to_post_map.get(node_id) for node_id in tree_nodes]
        all_posts.extend(posts)
        
    return all_posts