"""
Cross-Platform Analyzer Module

Refactored from cross_platform_analysis.py to be a proper module.
Handles cross-platform similarity matching and narrative analysis.
"""

import os
import sys
import json
import logging
import warnings
import time
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import re

# Import internal modules
from .similarity_graph import SimilarityGraphBuilder
from .narrative_classification import NarrativeClassifier
from .vllm_wrapper import create_vllm_pipeline, check_vllm_availability
from .text_processing import detect_post_language as detect_language

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CrossPlatformAnalyzer:
    """Cross-platform narrative analysis with similarity matching."""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = "cross_platform_output"):
        """Initialize the cross-platform analyzer."""
        # Process configuration
        if config:
            self.config = self._flatten_config(config)
        else:
            self.config = self._default_config()
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.platform_data = {}
        self.processed_data = {}
        self.similarity_graph = None
        self.matched_components = []
        self.narrative_results = []
        self.vllm_pipeline = None
        self.VLLM_AVAILABLE = False
        
        logger.info(f"Initialized CrossPlatformAnalyzer with output directory: {self.output_dir}")
    
    def _flatten_config(self, config: Dict) -> Dict:
        """Flatten nested configuration structure."""
        flat_config = {}
        
        # Data settings
        if 'data' in config:
            data_config = config['data']
            flat_config['data_dir'] = data_config.get('data_dir', 'data/data')
            flat_config['platform_files'] = {
                'truth': data_config.get('platforms', {}).get('truthsocial', 'truthsocial.trump.json'),
                'bluesky': data_config.get('platforms', {}).get('bluesky', 'bsky.trump.json'),
                'mastodon': data_config.get('platforms', {}).get('mastodon', 'mastodon.trump.json')
            }
            flat_config['max_posts_per_platform'] = data_config.get('max_posts_per_platform')
            flat_config['min_post_length'] = data_config.get('min_text_length', 10)
            flat_config['language_filter'] = data_config.get('language_filter', 'en')
        
        # Similarity settings
        if 'similarity' in config:
            sim_config = config['similarity']
            flat_config['embedding_model'] = sim_config.get('model_name', 'all-MiniLM-L6-v2')
            flat_config['similarity_threshold'] = sim_config.get('threshold', 0.7)
        
        # Visualization settings
        if 'embedding' in config:
            embed_config = config['embedding']
            flat_config['visualization'] = {
                'max_posts_for_tsne': embed_config.get('sample_size', 2000),
                'figure_dpi': 300,
                'figure_size': (12, 8)
            }
        else:
            flat_config['visualization'] = {
                'max_posts_for_tsne': 2000,
                'figure_dpi': 300,
                'figure_size': (12, 8)
            }
        
        return flat_config
    
    def _default_config(self) -> Dict:
        """Default configuration settings."""
        return {
            'data_dir': 'data/data',
            'platform_files': {
                'truth': 'truthsocial.trump.json',
                'bluesky': 'bsky.trump.json', 
                'mastodon': 'mastodon.trump.json'
            },
            'similarity_threshold': 0.65,
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_posts_per_platform': None,
            'min_post_length': 10,
            'language_filter': 'en',
            'visualization': {
                'max_posts_for_tsne': 2000,
                'figure_dpi': 300,
                'figure_size': (12, 8)
            }
        }
    
    def setup_vllm_pipeline(self) -> bool:
        """Setup vLLM Pipeline for narrative classification."""
        logger.info("Setting up vLLM Pipeline with Gemma 3 27B...")
        
        try:
            if not check_vllm_availability():
                logger.error("❌ vLLM not available")
                return False
            
            logger.info("Initializing vLLM pipeline...")
            self.vllm_pipeline = create_vllm_pipeline(
                model_name="google/gemma-3-27b-it",
                tensor_parallel_size=2,
                gpu_memory_utilization=0.8
            )
            
            logger.info("✅ vLLM Gemma 3 27B loaded successfully!")
            self.VLLM_AVAILABLE = True
            return True
                
        except Exception as e:
            logger.error(f"❌ vLLM pipeline setup failed: {e}")
            return False
    
    def load_platform_data(self) -> bool:
        """Load data from all platforms."""
        logger.info("Loading platform data...")
        
        data_dir = self.config['data_dir']
        platform_files = self.config['platform_files']
        
        for platform, filename in platform_files.items():
            filepath = os.path.join(data_dir, filename)
            logger.info(f"Loading {platform} data from {filepath}")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.platform_data[platform] = data
                logger.info(f"  - Loaded {len(data)} posts from {platform}")
            except FileNotFoundError:
                logger.warning(f"  - Warning: File {filepath} not found")
                self.platform_data[platform] = []
            except Exception as e:
                logger.error(f"  - Error loading {platform}: {e}")
                self.platform_data[platform] = []
        
        total_platforms = len([p for p in self.platform_data.values() if p])
        logger.info(f"Total platforms loaded: {total_platforms}")
        return total_platforms > 0
    
    def preprocess_post_content(self, content: str, platform: str = 'generic') -> str:
        """Preprocess post content for different platforms."""
        if platform == 'mastodon' or platform == 'truth':
            # Parse HTML content for Mastodon
            soup = BeautifulSoup(content, "html.parser")
            # Remove all 'a' tags entirely
            for a_tag in soup.find_all("a"):
                a_tag.unwrap()
            content = soup.get_text()
        
        # Common preprocessing for all platforms
        # Remove URLs
        content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
        # Remove @mentions
        content = re.sub(r'@\w+', '', content)
        # Remove hashtags (keep the text, remove #)
        content = re.sub(r'#(\w+)', r'\1', content)
        # Remove extra whitespace
        content = ' '.join(content.split())
        
        return content.strip()
    
    def extract_posts_from_platform_data(self) -> bool:
        """Extract and preprocess posts from platform data."""
        logger.info("Processing platform posts...")
        
        min_length = self.config['min_post_length']
        language_filter = self.config['language_filter']
        
        for platform, data in self.platform_data.items():
            all_posts = []
            all_metadata = []
            root_posts = []
            root_metadata = []
            
            logger.info(f"Processing {platform} posts...")
            
            for post in tqdm(data, desc=f"Processing {platform}"):
                try:
                    # Extract content based on platform structure
                    if platform == 'mastodon':
                        content = post.get('content', '')
                        is_reply = bool(post.get('in_reply_to_id'))
                    elif platform == 'truth':
                        content = post.get('content', '')
                        is_reply = bool(post.get('in_reply_to_id'))
                    elif platform == 'bluesky':
                        content = post.get('record', {}).get('text', '')
                        is_reply = 'reply' in post.get('record', {}).get('$type', '')
                    else:
                        content = post.get('text', '') or post.get('content', '')
                        is_reply = False
                    
                    # Preprocess content
                    processed_content = self.preprocess_post_content(content, platform)
                    
                    # Filter out very short posts
                    if len(processed_content) < min_length:
                        continue
                    
                    # Filter by language if specified
                    if language_filter:
                        try:
                            if detect_language(processed_content).lower() != language_filter:
                                continue
                        except:
                            pass
                    
                    # Store metadata for all posts
                    metadata = {
                        'platform': platform,
                        'is_reply': is_reply,
                        'original_content': content,
                        'processed_content': processed_content,
                        'post_id': post.get('id', len(all_metadata)),
                        'created_at': post.get('created_at', ''),
                        'author_id': post.get('account', {}).get('id', '') if platform == 'mastodon' else post.get('author_id', ''),
                        'original_post': post
                    }
                    
                    all_posts.append(processed_content)
                    all_metadata.append(metadata)
                    
                    # For matching, only use root posts (non-replies)
                    if not is_reply:
                        root_posts.append(processed_content)
                        root_metadata.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Error processing post from {platform}: {e}")
                    continue
            
            # Store both all posts and root posts separately
            self.processed_data[platform] = {
                'all_posts': all_posts,
                'all_metadata': all_metadata,
                'posts': root_posts,
                'metadata': root_metadata,
                'stats': {
                    'total_posts': len(all_posts),
                    'root_posts': len(root_posts),
                    'reply_posts': len(all_posts) - len(root_posts)
                }
            }
            
            logger.info(f"  - {platform}: {len(all_posts)} total posts ({len(root_posts)} root, {len(all_posts) - len(root_posts)} replies)")
        
        # Save intermediate data
        self._save_intermediate_data('processed_posts', self.processed_data)
        
        return len(self.processed_data) > 0
    
    def create_summary_statistics(self) -> pd.DataFrame:
        """Create summary statistics for loaded data."""
        logger.info("Creating summary statistics...")
        
        summary_data = []
        for platform, data in self.processed_data.items():
            stats = data['stats']
            root_posts = data['posts']
            all_posts = data['all_posts']
            
            root_avg_length = np.mean([len(post.split()) for post in root_posts]) if root_posts else 0
            all_avg_length = np.mean([len(post.split()) for post in all_posts]) if all_posts else 0
            
            summary_data.append({
                'Platform': platform.title(),
                'Total Posts': stats['total_posts'],
                'Root Posts': stats['root_posts'],
                'Reply Posts': stats['reply_posts'],
                'Root Posts %': f"{(stats['root_posts'] / stats['total_posts'] * 100):.1f}%" if stats['total_posts'] > 0 else "0%",
                'Avg Root Length': f"{root_avg_length:.1f}",
                'Avg All Length': f"{all_avg_length:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info("Platform Data Summary:")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        
        # Save summary to file
        summary_df.to_csv(self.output_dir / 'platform_summary.csv', index=False)
        self._save_intermediate_data('summary_statistics', summary_df, 'csv')
        
        return summary_df
    
    def build_similarity_graph(self) -> bool:
        """Build cross-platform similarity graph using only root posts."""
        logger.info("Building cross-platform similarity graph...")
        
        # Prepare data for similarity graph - using only root posts
        platform_posts_only = {}
        for platform, data in self.processed_data.items():
            platform_posts_only[platform] = data['posts']
        
        # Filter platforms with sufficient data
        filtered_platforms = {k: v for k, v in platform_posts_only.items() if len(v) >= 10}
        logger.info(f"Using platforms with >=10 root posts: {list(filtered_platforms.keys())}")
        
        if len(filtered_platforms) < 2:
            logger.error("Not enough platforms with sufficient root posts for similarity matching.")
            return False
        
        # Build similarity graph
        similarity_builder = SimilarityGraphBuilder(
            model_name=self.config['embedding_model'], 
            similarity_threshold=self.config['similarity_threshold']
        )
        
        if len(filtered_platforms) == 3:
            # Use tripartite graph for 3 platforms
            logger.info("Building tripartite similarity graph for 3 platforms...")
            self.similarity_graph, self.offsets = similarity_builder.build_tripartite_graph(filtered_platforms)
            logger.info(f"Tripartite graph created with {self.similarity_graph.number_of_nodes()} nodes and {self.similarity_graph.number_of_edges()} edges")
        else:
            # For 2 platforms, build bipartite graph
            logger.info("Building bipartite graph for 2 platforms...")
            platforms_list = list(filtered_platforms.keys())
            
            # Get embeddings for both platforms
            embeddings = {}
            for platform, posts in filtered_platforms.items():
                logger.info(f"Encoding {len(posts)} root posts from {platform}...")
                embeddings[platform] = similarity_builder.encode_posts(posts)
            
            # Compute similarity between the two platforms
            platform_a, platform_b = platforms_list[0], platforms_list[1]
            sim_matrix = similarity_builder.compute_similarity(
                embeddings[platform_a], 
                embeddings[platform_b]
            )
            
            logger.info(f"Similarity matrix shape: {sim_matrix.shape}")
            matches_above_threshold = (sim_matrix > similarity_builder.threshold).sum()
            logger.info(f"Matches above threshold ({similarity_builder.threshold}): {matches_above_threshold}")
            
            # Create a simple graph for visualization
            self.similarity_graph = nx.Graph()
            
            # Add nodes
            for i, platform in enumerate([platform_a, platform_b]):
                num_posts = len(filtered_platforms[platform])
                offset = i * 10000
                self.similarity_graph.add_nodes_from(
                    range(offset, offset + num_posts),
                    platform=platform
                )
            
            # Add edges for similar posts
            threshold = similarity_builder.threshold
            rows, cols = np.where(sim_matrix.cpu().numpy() > threshold)
            edges_added = 0
            for row, col in zip(rows, cols):
                node_a = 0 + row
                node_b = 10000 + col
                similarity_score = float(sim_matrix[row, col])
                self.similarity_graph.add_edge(node_a, node_b, weight=similarity_score)
                edges_added += 1
            
            self.offsets = {platform_a: 0, platform_b: 10000}
            
            logger.info(f"Bipartite graph created with {self.similarity_graph.number_of_nodes()} nodes and {edges_added} edges")
        
        return True
    
    def extract_matched_components(self) -> bool:
        """Extract connected components (matched post clusters)."""
        if self.similarity_graph is None:
            logger.error("No similarity graph available for component extraction.")
            return False
        
        logger.info("Extracting connected components from root post matches...")
        
        # Get filtered platforms data (root posts only)
        filtered_platforms = {k: v for k, v in {platform: data['posts'] for platform, data in self.processed_data.items()}.items() if len(v) >= 10}
        
        # Create a temporary similarity builder instance for component extraction
        temp_builder = SimilarityGraphBuilder()
        self.matched_components = temp_builder.extract_connected_components(
            self.similarity_graph, filtered_platforms, self.offsets
        )
        
        logger.info(f"Found {len(self.matched_components)} connected components (matched clusters)")
        
        # Save components to file
        components_data = []
        for component in self.matched_components:
            component_data = {}
            for platform, posts in component.items():
                component_data[platform] = [
                    {
                        'content': post,
                        'metadata': self.processed_data[platform]['metadata'][idx] if idx < len(self.processed_data[platform]['metadata']) else {}
                    }
                    for idx, post in posts
                ]
            components_data.append(component_data)
        
        with open(self.output_dir / 'matched_components.json', 'w') as f:
            json.dump(components_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    def analyze_narrative_frames(self) -> bool:
        """Analyze narrative frames in matched components."""
        logger.info("Analyzing narrative frames...")
        
        if not self.matched_components:
            logger.error("No matched components available for narrative analysis.")
            return False
        
        # Initialize narrative classifier
        if self.VLLM_AVAILABLE and self.vllm_pipeline:
            narrative_classifier = NarrativeClassifier(llm_pipeline=self.vllm_pipeline)
        else:
            logger.warning("vLLM not available, using demo classifier")
            narrative_classifier = NarrativeClassifier()
        
        # Analyze each component
        for i, component in enumerate(tqdm(self.matched_components[:100], desc="Analyzing narratives")):  # Limit for testing
            component_analysis = {
                'component_id': i,
                'platforms': {}
            }
            
            for platform, posts in component.items():
                platform_results = []
                for post_idx, post_content in posts:
                    try:
                        classification = narrative_classifier.classify_post(post_content)
                        platform_results.append({
                            'post_content': post_content,
                            'post_index': post_idx,
                            'narrative_frames': classification
                        })
                    except Exception as e:
                        logger.error(f"Error classifying post: {e}")
                        continue
                
                component_analysis['platforms'][platform] = platform_results
            
            self.narrative_results.append(component_analysis)
        
        # Save narrative analysis results
        with open(self.output_dir / 'narrative_analysis_results.json', 'w') as f:
            json.dump(self.narrative_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analyzed narrative frames for {len(self.narrative_results)} components")
        return True
    
    def create_narrative_visualizations(self) -> bool:
        """Create visualizations for narrative analysis."""
        logger.info("Creating narrative visualizations...")
        
        if not self.narrative_results:
            logger.warning("No narrative results available for visualization")
            return False
        
        try:
            # Create basic visualization placeholder
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Narrative Analysis Visualizations\n(Implementation needed)", 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'narrative_visualizations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Basic visualization created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return False
    
    def _save_intermediate_data(self, name: str, data: Any, format_type: str = 'json'):
        """Save intermediate data to file."""
        try:
            if format_type == 'json':
                filepath = self.output_dir / f'{name}.json'
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format_type == 'csv' and isinstance(data, pd.DataFrame):
                filepath = self.output_dir / f'{name}.csv'
                data.to_csv(filepath, index=False)
            
            logger.debug(f"Saved intermediate data: {filepath}")
        except Exception as e:
            logger.error(f"Error saving intermediate data {name}: {e}")
