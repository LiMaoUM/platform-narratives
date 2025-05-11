import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

class SimilarityGraphBuilder:
    """
    Builds similarity graphs between posts from different platforms based on semantic similarity.
    This enables cross-platform narrative analysis by connecting similar content.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.7):
        """
        Initialize the similarity graph builder.
        
        Args:
            model_name: The name of the sentence transformer model to use
            similarity_threshold: Threshold for considering two posts as similar (0-1)
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
    
    def encode_posts(self, posts_list, batch_size=1024, show_progress=True):
        """
        Encode posts into embeddings.
        
        Args:
            posts_list: List of post texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Tensor of embeddings
        """
        return self.model.encode(
            posts_list, 
            convert_to_tensor=True, 
            show_progress_bar=show_progress, 
            batch_size=batch_size
        )
    
    def compute_similarity(self, embeddings_a, embeddings_b):
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings_a: First set of embeddings
            embeddings_b: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        return util.pytorch_cos_sim(embeddings_a.cpu(), embeddings_b.cpu())
    
    def build_tripartite_graph(self, platform_data):
        """
        Build a tripartite graph connecting similar posts across three platforms.
        
        Args:
            platform_data: Dictionary with keys as platform names and values as lists of posts
                Example: {'truth': [...], 'bluesky': [...], 'mastodon': [...]}  
                
        Returns:
            NetworkX graph and node offset information
        """
        # Extract platform names and posts
        platforms = list(platform_data.keys())
        if len(platforms) != 3:
            raise ValueError("Tripartite graph requires exactly 3 platforms")
        
        # Encode posts for each platform
        embeddings = {}
        for platform, posts in platform_data.items():
            print(f"Encoding {len(posts)} posts from {platform}...")
            embeddings[platform] = self.encode_posts(posts)
        
        # Calculate offsets to avoid node ID collisions
        offsets = {}
        current_offset = 0
        for platform in platforms:
            offsets[platform] = current_offset
            current_offset += len(embeddings[platform])
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes with platform labels
        for platform in platforms:
            offset = offsets[platform]
            platform_embeddings = embeddings[platform]
            G.add_nodes_from(
                range(offset, offset + len(platform_embeddings)), 
                platform=platform
            )
        
        # Compute similarities and add edges between all platform pairs
        for i in range(len(platforms)):
            for j in range(i+1, len(platforms)):
                platform_a = platforms[i]
                platform_b = platforms[j]
                
                print(f"Computing similarity between {platform_a} and {platform_b}...")
                sim_matrix = self.compute_similarity(
                    embeddings[platform_a], 
                    embeddings[platform_b]
                )
                
                # Add edges for similar posts
                edges = self._get_edges_from_similarity(
                    sim_matrix, 
                    offsets[platform_a], 
                    offsets[platform_b]
                )
                G.add_edges_from(edges)
        
        print(f"Tripartite graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G, offsets
    
    def _get_edges_from_similarity(self, sim_matrix, offset_a, offset_b):
        """
        Helper to add weighted edges from similarity matrix.
        
        Args:
            sim_matrix: Similarity matrix
            offset_a: Node ID offset for first platform
            offset_b: Node ID offset for second platform
            
        Returns:
            List of edges with weights
        """
        row, col = np.where(sim_matrix > self.threshold)
        weights = sim_matrix[row, col]
        edges = [
            (offset_a + i, offset_b + j, {"weight": float(w)})
            for i, j, w in zip(row, col, weights)
        ]
        return edges
    
    def extract_connected_components(self, graph, platform_posts, offsets):
        """
        Extract connected components from the graph and organize posts by platform.
        
        Args:
            graph: NetworkX graph of connected posts
            platform_posts: Dictionary mapping platform names to post lists
            offsets: Dictionary mapping platform names to node ID offsets
            
        Returns:
            List of dictionaries, each containing posts from different platforms in a component
        """
        components = list(nx.connected_components(graph))
        # Filter to components with more than one node
        components = [c for c in components if len(c) > 1]
        
        component_posts = []
        for component in components:
            posts = {platform: [] for platform in platform_posts.keys()}
            
            for node in component:
                for platform, offset in offsets.items():
                    next_platform_offset = offset + len(platform_posts[platform])
                    if offset <= node < next_platform_offset:
                        post_idx = node - offset
                        posts[platform].append(platform_posts[platform][post_idx])
                        break
            
            component_posts.append(posts)
        
        return component_posts