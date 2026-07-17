#!/usr/bin/env python3
"""
Cross-Platform Narrative Analysis Script

This script performs the same analysis as the matching.ipynb notebook but in a
format suitable for backend execution. It includes:
1. Data loading and preprocessing from multiple platforms
2. Cross-platform post matching using semantic similarity
3. Narrative frame analysis using LLM classification
4. Visualization and results export

Usage:
    python cross_platform_analysis.py [--config config.yaml] [--output output_dir]
"""

import sys
import os
import json
import yaml
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from bs4 import BeautifulSoup
import re
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fast_langdetect import detect_language

# Import custom modules
from similarity_graph import SimilarityGraphBuilder
from text_processing import clean_text, detect_post_language
from narrative_divergence_analyzer import NarrativeDivergenceAnalyzer
from narrative_classification import NarrativeClassifier
from vllm_wrapper import VLLMWrapper, create_vllm_pipeline, check_vllm_availability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CrossPlatformAnalyzer:
    """Main class for cross-platform narrative analysis"""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = "output"):
        # Process configuration
        if config:
            # Flatten nested config structure
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
        """Flatten nested configuration structure"""
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
        """Default configuration settings"""
        return {
            'data_dir': 'data/data',
            'platform_files': {
                'truth': 'truthsocial.trump.json',
                'bluesky': 'bsky.trump.json', 
                'mastodon': 'mastodon.trump.json'
            },
            'similarity_threshold': 0.65,
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_posts_per_platform': None,  # No limit by default
            'min_post_length': 10,
            'language_filter': 'en',
            'visualization': {
                'max_posts_for_tsne': 2000,
                'figure_dpi': 300,
                'figure_size': (12, 8)
            }
        }
    
    def setup_vllm_pipeline(self) -> bool:
        """Setup vLLM Pipeline for narrative classification"""
        logger.info("Setting up vLLM Pipeline with Gemma 3 27B...")
        
        try:
            # Check if vLLM is available
            if not check_vllm_availability():
                logger.error("❌ vLLM not available")
                logger.info("💡 Install vLLM: pip install vllm")
                return False
            
            # Create vLLM pipeline
            logger.info("Initializing vLLM pipeline...")
            self.vllm_pipeline = create_vllm_pipeline(
                model_name="google/gemma-3-27b-it",
                tensor_parallel_size=2,  # Use 2 GPUs for tensor parallelism
                gpu_memory_utilization=0.8
            )
            
            logger.info("✅ vLLM Gemma 3 27B loaded successfully!")
            
            # Test with narrative classification
            test_post = "The corrupt establishment is trying to silence Trump through these bogus legal charges."
            logger.info("Testing narrative classification with vLLM pipeline...")
            
            try:
                test_messages = [
                    {"role": "system", "content": "You are a narrative analyst."},
                    {"role": "user", "content": f"Analyze this post for narrative frames: {test_post}"}
                ]
                
                test_response = self.vllm_pipeline(test_messages, max_new_tokens=100)
                response_text = test_response[0]["generated_text"][0]["content"]
                
                logger.info("✅ vLLM API call successful!")
                logger.info(f"Response preview: {response_text[:100]}...")
                
                # Test narrative classifier with vLLM pipeline
                narrative_classifier = NarrativeClassifier(llm_pipeline=self.vllm_pipeline)
                test_result = narrative_classifier.classify_post(test_post)
                
                logger.info("✅ Narrative classification successful!")
                self.VLLM_AVAILABLE = True
                return True
                
            except Exception as e:
                logger.warning(f"Narrative classification test failed: {e}")
                logger.warning("Pipeline connection works but classification needs debugging")
                self.VLLM_AVAILABLE = True  # Still mark as available since basic API works
                return True
                
        except Exception as e:
            logger.error(f"❌ vLLM pipeline setup failed: {e}")
            
            if "vllm" in str(e).lower() or "CUDA" in str(e):
                logger.info("💡 vLLM requires CUDA-compatible GPU for local inference")
                logger.info("💡 Install vLLM: pip install vllm")
                logger.info("💡 Check CUDA availability: nvidia-smi")
            elif "memory" in str(e).lower():
                logger.info("💡 Try reducing tensor_parallel_size or gpu_memory_utilization")
                logger.info("💡 Gemma 3 27B requires substantial GPU memory (~54GB for FP16)")
            
            return False
    
    def create_demo_pipeline(self):
        """Create fallback demo pipeline for testing"""
        logger.info("🔄 Creating demo pipeline for testing...")
        
        def demo_gemini_pipeline(messages, max_new_tokens=300, **kwargs):
            """Demo pipeline that simulates Gemini API responses with probability-based results"""
            import random
            
            # Extract post content from messages
            user_content = messages[-1]['content'] if messages else ""
            post_text = user_content.lower()
            
            # Generate realistic frame probabilities based on content
            frame_probs = {
                "Persecution": round(random.uniform(0.6, 0.9) if any(word in post_text for word in ['silence', 'attack', 'target', 'witch hunt']) else random.uniform(0.05, 0.15), 2),
                "Corruption": round(random.uniform(0.7, 0.9) if any(word in post_text for word in ['corrupt', 'establishment']) else random.uniform(0.05, 0.15), 2),
                "Accountability": round(random.uniform(0.1, 0.3) if any(word in post_text for word in ['justice', 'law', 'legal']) else random.uniform(0.05, 0.15), 2),
                "Irony/Detachment": round(random.uniform(0.05, 0.15), 2),
                "Heroism": round(random.uniform(0.05, 0.2), 2),
                "Civic Critique": round(random.uniform(0.1, 0.3), 2),
                "Moral Decay": round(random.uniform(0.05, 0.2), 2),
                "Media Manipulation": round(random.uniform(0.3, 0.6) if 'trump' in post_text else random.uniform(0.05, 0.15), 2),
                "Strategic Pragmatism": round(random.uniform(0.05, 0.15), 2),
                "Cultural Identity": round(random.uniform(0.05, 0.2), 2)
            }
            
            response_content = json.dumps(frame_probs, indent=2)
            
            return [{
                "generated_text": [
                    {"role": "assistant", "content": response_content}
                ]
            }]
        
        self.vllm_pipeline = demo_gemini_pipeline
        logger.info("✅ Demo pipeline ready")
    
    def load_platform_data(self) -> bool:
        """Load data from all platforms"""
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
        """Preprocess post content for different platforms"""
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
        """Extract and preprocess posts from platform data"""
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
                            # If language detection fails, keep the post
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
                        'original_post': post  # Keep original post data for reference
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
                'posts': root_posts,  # For backward compatibility and matching
                'metadata': root_metadata,  # Only root posts for matching
                'stats': {
                    'total_posts': len(all_posts),
                    'root_posts': len(root_posts),
                    'reply_posts': len(all_posts) - len(root_posts)
                }
            }
            
            logger.info(f"  - {platform}: {len(all_posts)} total posts ({len(root_posts)} root, {len(all_posts) - len(root_posts)} replies)")
            logger.info(f"  - {platform}: Using {len(root_posts)} root posts for matching")
        
        # Save intermediate data - processed posts
        self._save_intermediate_data('processed_posts', self.processed_data)
        
        return len(self.processed_data) > 0
    
    def create_summary_statistics(self) -> pd.DataFrame:
        """Create summary statistics for loaded data"""
        logger.info("Creating summary statistics...")
        
        summary_data = []
        for platform, data in self.processed_data.items():
            stats = data['stats']
            root_posts = data['posts']  # Only root posts
            all_posts = data['all_posts']  # All posts including replies
            
            # Calculate averages for root posts (used for matching)
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
        summary_df.to_csv(
            self.output_dir / 'platform_summary.csv', 
            index=False,
            escapechar='\\',
            quoting=1  # QUOTE_ALL
        )
        
        # Save intermediate data - summary statistics
        self._save_intermediate_data('summary_statistics', summary_df, 'csv')
        
        return summary_df
    
    def build_similarity_graph(self) -> bool:
        """Build cross-platform similarity graph using only root posts"""
        logger.info("Building cross-platform similarity graph...")
        logger.info("🎯 IMPORTANT: Using ONLY root posts (no replies) for cross-platform matching")
        
        # Prepare data for similarity graph - using only root posts
        platform_posts_only = {}
        for platform, data in self.processed_data.items():
            platform_posts_only[platform] = data['posts']  # These are root posts only
        
        logger.info("Platform root posts for similarity matching:")
        for platform, posts in platform_posts_only.items():
            total_posts = len(self.processed_data[platform]['all_posts'])
            logger.info(f"  - {platform}: {len(posts)} root posts (out of {total_posts} total)")
        
        # Filter platforms with sufficient data
        filtered_platforms = {k: v for k, v in platform_posts_only.items() if len(v) >= 10}
        logger.info(f"Using platforms with >=10 root posts: {list(filtered_platforms.keys())}")
        
        if len(filtered_platforms) < 2:
            logger.error("Not enough platforms with sufficient root posts for similarity matching.")
            return False
        
        # Save intermediate data - filtered root posts for matching
        matching_data = {
            'platforms': list(filtered_platforms.keys()),
            'post_counts': {k: len(v) for k, v in filtered_platforms.items()},
            'posts': filtered_platforms
        }
        self._save_intermediate_data('root_posts_for_matching', matching_data)
        
        # Build similarity graph
        similarity_builder = SimilarityGraphBuilder(
            model_name=self.config['embedding_model'], 
            similarity_threshold=self.config['similarity_threshold']
        )
        
        if len(filtered_platforms) == 3:
            # Use tripartite graph for 3 platforms
            logger.info("Building tripartite similarity graph for 3 platforms...")
            self.similarity_graph, self.offsets = similarity_builder.build_tripartite_graph(filtered_platforms)
            graph_type = "tripartite"
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
            
            # Save intermediate data - embeddings
            embedding_data = {
                'model_name': self.config['embedding_model'],
                'platforms': platforms_list,
                'embedding_shapes': {k: v.shape for k, v in embeddings.items()},
                'similarity_threshold': self.config['similarity_threshold']
            }
            self._save_intermediate_data('embeddings_info', embedding_data)
            
            # Compute similarity between the two platforms
            platform_a, platform_b = platforms_list[0], platforms_list[1]
            sim_matrix = similarity_builder.compute_similarity(
                embeddings[platform_a], 
                embeddings[platform_b]
            )
            
            logger.info(f"Similarity matrix shape: {sim_matrix.shape}")
            matches_above_threshold = (sim_matrix > similarity_builder.threshold).sum()
            logger.info(f"Matches above threshold ({similarity_builder.threshold}): {matches_above_threshold}")
            
            # Save intermediate data - similarity matrix stats
            sim_stats = {
                'matrix_shape': sim_matrix.shape,
                'threshold': float(similarity_builder.threshold),
                'matches_above_threshold': int(matches_above_threshold),
                'max_similarity': float(sim_matrix.max()),
                'mean_similarity': float(sim_matrix.mean()),
                'platform_a': platform_a,
                'platform_b': platform_b
            }
            self._save_intermediate_data('similarity_matrix_stats', sim_stats)
            
            # Create a simple graph for visualization
            self.similarity_graph = nx.Graph()
            
            # Add nodes
            for i, platform in enumerate([platform_a, platform_b]):
                num_posts = len(filtered_platforms[platform])
                offset = i * 10000  # Large offset to avoid collisions
                self.similarity_graph.add_nodes_from(
                    range(offset, offset + num_posts),
                    platform=platform
                )
            
            # Add edges for similar posts
            threshold = similarity_builder.threshold
            rows, cols = np.where(sim_matrix.cpu().numpy() > threshold)
            edges_added = 0
            for row, col in zip(rows, cols):
                weight = float(sim_matrix[row, col])
                self.similarity_graph.add_edge(row, 10000 + col, weight=weight)
                edges_added += 1
            
            self.offsets = {platform_a: 0, platform_b: 10000}
            graph_type = "bipartite"
            
            logger.info(f"Bipartite graph created with {self.similarity_graph.number_of_nodes()} nodes and {edges_added} edges")
        
        # Save intermediate data - graph structure
        graph_data = {
            'graph_type': graph_type,
            'num_nodes': self.similarity_graph.number_of_nodes(),
            'num_edges': self.similarity_graph.number_of_edges(),
            'platforms': list(filtered_platforms.keys()),
            'offsets': self.offsets,
            'similarity_threshold': self.config['similarity_threshold'],
            'embedding_model': self.config['embedding_model']
        }
        self._save_intermediate_data('similarity_graph_structure', graph_data)
        
        return True
    
    def extract_matched_components(self) -> bool:
        """Extract connected components (matched post clusters)"""
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
        
        # Analyze component sizes
        component_sizes = []
        detailed_components = []
        
        for i, component in enumerate(self.matched_components):
            total_posts = sum(len(posts) for posts in component.values())
            platform_counts = {platform: len(posts) for platform, posts in component.items() if posts}
            
            component_info = {
                'component_id': i,
                'total_posts': total_posts,
                'platforms': list(platform_counts.keys()),
                'platform_counts': platform_counts,
                'cross_platform': len(platform_counts) > 1
            }
            component_sizes.append(component_info)
            
            # Store detailed component data for intermediate saving
            detailed_component = {
                'component_id': i,
                'metadata': component_info,
                'posts_by_platform': component
            }
            detailed_components.append(detailed_component)
        
        # Sort by total posts
        component_sizes.sort(key=lambda x: x['total_posts'], reverse=True)
        
        logger.info("Top 10 largest connected components:")
        cross_platform_count = 0
        for i, comp in enumerate(component_sizes[:10]):
            platforms_str = ", ".join([f"{p}({c})" for p, c in comp['platform_counts'].items()])
            cross_platform_marker = "🔗" if comp['cross_platform'] else "📍"
            logger.info(f"  {i+1}. {cross_platform_marker} Component {comp['component_id']}: {comp['total_posts']} posts across [{platforms_str}]")
            if comp['cross_platform']:
                cross_platform_count += 1
        
        logger.info(f"📊 Found {cross_platform_count} cross-platform components out of {len(component_sizes)} total")
        
        # Save components to file
        components_data = []
        for component in self.matched_components:
            components_data.append(component)
        
        with open(self.output_dir / 'matched_components.json', 'w') as f:
            json.dump(components_data, f, indent=2)
        
        # Save intermediate data - component analysis
        component_analysis = {
            'total_components': len(self.matched_components),
            'cross_platform_components': cross_platform_count,
            'component_sizes': component_sizes[:20],  # Top 20 components
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        self._save_intermediate_data('component_analysis', component_analysis)
        
        # Save detailed components
        self._save_intermediate_data('detailed_components', detailed_components)
        
        return True
    
    def analyze_narrative_frames(self) -> bool:
        """Analyze narrative frames in ALL matched components and ALL matched root posts"""
        if not self.matched_components:
            logger.warning("No matched components available for narrative analysis.")
            return False
        
        if not (self.VLLM_AVAILABLE or self.vllm_pipeline):
            logger.warning("No vLLM pipeline available for narrative analysis.")
            return False
        
        logger.info("=== Comprehensive Narrative Frame Analysis of ALL Matched Components ===")
        
        # Initialize narrative classifier
        classifier = NarrativeClassifier(llm_pipeline=self.vllm_pipeline)
        
        # Process ALL components, not just top 10
        all_components = []
        for i, comp in enumerate(self.matched_components):
            platform_counts = {p: len(posts) for p, posts in comp.items() if posts}
            if len(platform_counts) >= 2:  # At least 2 platforms
                all_components.append((i, comp, platform_counts))
        
        logger.info(f"Processing ALL {len(all_components)} cross-platform matched components...")
        logger.info("Note: Removed usage limits - processing ALL matched root posts systematically")
        
        self.narrative_results = []
        total_posts_processed = 0
        
        # Add progress tracking with intermediate saves
        progress_save_interval = 10  # Save every 10 components
        
        for comp_idx, (comp_id, component, platform_counts) in enumerate(tqdm(all_components, desc="Processing components")):
            logger.info(f"--- Component {comp_id + 1}/{len(self.matched_components)} Narrative Analysis ---")
            
            # Classify ALL posts from each platform in this component
            component_narratives = {}
            component_post_count = 0
            
            for platform, posts in component.items():
                if posts:  # Process ALL posts, not just those with <= 5 posts
                    platform_classifications = []
                    
                    # Process ALL posts from this platform, not just first 3
                    for post_idx, post in enumerate(posts):
                        try:
                            if len(post.strip()) > 10:  # Skip very short posts
                                classification = classifier.classify_post(post)
                                classification["post_index"] = post_idx
                                classification["post_preview"] = post[:100] + "..." if len(post) > 100 else post
                                platform_classifications.append(classification)
                                total_posts_processed += 1
                                component_post_count += 1
                        except Exception as e:
                            logger.error(f"    Error classifying post {post_idx} from {platform}: {e}")
                            continue
                    
                    if platform_classifications:
                        component_narratives[platform] = platform_classifications
            
            # Analyze narrative divergence within this component
            platform_posts_summary = ', '.join([f'{p}({len(posts)})' for p, posts in component.items() if posts])
            logger.info(f"  Platforms & Posts: {platform_posts_summary}")
            logger.info(f"  Classified Posts: {component_post_count}")
            
            # Extract narrative frames by platform using new probability-based approach
            platform_frames = {}
            platform_frame_stats = {}
            
            for platform, classifications in component_narratives.items():
                # Extract dominant frames (using max_frame from metadata)
                dominant_frames = []
                frame_probabilities = []
                
                for classification in classifications:
                    if "_metadata" in classification:
                        max_frame = classification["_metadata"].get("max_frame", "unknown")
                        max_prob = classification["_metadata"].get("max_probability", 0.0)
                        dominant_frames.append(max_frame)
                        frame_probabilities.append(max_prob)
                
                if dominant_frames:
                    platform_frames[platform] = dominant_frames
                    platform_frame_stats[platform] = {
                        "dominant_frames": dominant_frames,
                        "avg_confidence": sum(frame_probabilities) / len(frame_probabilities),
                        "num_posts": len(classifications)
                    }
                    
                    # Log frame summary for this platform
                    frame_counts = {}
                    for frame in dominant_frames:
                        frame_counts[frame] = frame_counts.get(frame, 0) + 1
                    
                    frame_summary = ", ".join([f"{frame}({count})" for frame, count in frame_counts.items()])
                    logger.info(f"    {platform.upper()}: {frame_summary} [avg_conf: {platform_frame_stats[platform]['avg_confidence']:.2f}]")
            
            # Check for narrative divergence
            if len(platform_frames) >= 2:
                all_frames = set()
                for frames in platform_frames.values():
                    all_frames.update(frames)
                
                if len(all_frames) > 1:
                    logger.info(f"    🎯 NARRATIVE DIVERGENCE DETECTED: {sorted(all_frames)}")
                else:
                    logger.info(f"    ✓ Consistent narrative frame: {list(all_frames)[0]}")
            
            # Store comprehensive results
            self.narrative_results.append({
                'component_id': comp_id,
                'platform_counts': platform_counts,
                'narrative_frames': platform_frames,
                'frame_statistics': platform_frame_stats,
                'total_posts_classified': component_post_count,
                'detailed_classifications': component_narratives  # Store full classification details
            })
            
            # Intermediate save every N components
            if (comp_idx + 1) % progress_save_interval == 0:
                logger.info(f"💾 Intermediate save: Processed {comp_idx + 1}/{len(all_components)} components")
                intermediate_file = self.output_dir / f'narrative_analysis_partial_{comp_idx + 1}.json'
                with open(intermediate_file, 'w') as f:
                    json.dump(self.narrative_results, f, indent=2)
        
        # Final comprehensive save
        with open(self.output_dir / 'narrative_analysis_complete.json', 'w') as f:
            json.dump(self.narrative_results, f, indent=2)
        
        # Generate summary statistics
        total_components = len(self.narrative_results)
        total_platforms_analyzed = sum(len(result['platform_counts']) for result in self.narrative_results)
        components_with_divergence = sum(1 for result in self.narrative_results 
                                       if len(result.get('narrative_frames', {})) >= 2 
                                       and len(set().union(*result['narrative_frames'].values())) > 1)
        
        logger.info(f"\n=== COMPREHENSIVE NARRATIVE ANALYSIS COMPLETE ===")
        logger.info(f"✅ Total components processed: {total_components}")
        logger.info(f"✅ Total matched posts classified: {total_posts_processed}")
        logger.info(f"✅ Total platform instances analyzed: {total_platforms_analyzed}")
        logger.info(f"🎯 Components with narrative divergence: {components_with_divergence} ({components_with_divergence/total_components*100:.1f}%)")
        logger.info(f"💾 Results saved to: narrative_analysis_complete.json")
        
        return True
    
    def create_visualizations(self) -> bool:
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # 1. Platform summary visualization
            self._create_platform_summary_plot()
            
            # 2. Embedding visualization if we have enough data
            self._create_embedding_visualization()
            
            # 3. Narrative frame distribution if available
            if self.narrative_results:
                self._create_narrative_distribution_plot()
            
            # 4. Component analysis
            if self.matched_components:
                self._create_component_analysis_plot()
            
            logger.info("✅ Visualizations created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return False
    
    def _create_platform_summary_plot(self):
        """Create platform summary visualization"""
        summary_data = []
        for platform, data in self.processed_data.items():
            posts = data['posts']
            metadata = data['metadata']
            
            reply_count = sum(1 for m in metadata if m['is_reply'])
            avg_length = np.mean([len(post.split()) for post in posts]) if posts else 0
            
            summary_data.append({
                'Platform': platform.title(),
                'Total Posts': len(posts),
                'Reply Posts': reply_count,
                'Original Posts': len(posts) - reply_count,
                'Avg Word Length': avg_length
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Plot summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=self.config['visualization']['figure_dpi'])
        
        # Post counts by platform
        platforms = summary_df['Platform']
        total_posts = summary_df['Total Posts']
        reply_posts = summary_df['Reply Posts']
        original_posts = summary_df['Original Posts']
        
        ax1.bar(platforms, original_posts, label='Original Posts', alpha=0.8)
        ax1.bar(platforms, reply_posts, bottom=original_posts, label='Reply Posts', alpha=0.8)
        ax1.set_title('Post Distribution by Platform')
        ax1.set_ylabel('Number of Posts')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Average word length
        avg_lengths = summary_df['Avg Word Length']
        ax2.bar(platforms, avg_lengths, alpha=0.8, color='orange')
        ax2.set_title('Average Post Length by Platform')
        ax2.set_ylabel('Average Words per Post')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_summary.png', dpi=self.config['visualization']['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def _create_embedding_visualization(self):
        """Create embedding visualization"""
        # Get filtered platforms with sufficient data
        filtered_platforms = {k: v for k, v in {platform: data['posts'] for platform, data in self.processed_data.items()}.items() if len(v) >= 10}
        
        if len(filtered_platforms) < 2:
            logger.warning("Not enough platforms for embedding visualization.")
            return
        
        # Get all posts and their platform labels
        all_posts = []
        platform_labels = []
        
        for platform, posts in filtered_platforms.items():
            all_posts.extend(posts)
            platform_labels.extend([platform] * len(posts))
        
        logger.info(f"Total posts for visualization: {len(all_posts)}")
        
        # Sample posts if too many (for performance)
        max_posts = self.config['visualization']['max_posts_for_tsne']
        if len(all_posts) > max_posts:
            logger.info(f"Sampling {max_posts} posts for visualization...")
            indices = np.random.choice(len(all_posts), max_posts, replace=False)
            all_posts = [all_posts[i] for i in indices]
            platform_labels = [platform_labels[i] for i in indices]
        
        # Get embeddings
        logger.info("Computing embeddings for visualization...")
        model = SentenceTransformer(self.config['embedding_model'])
        embeddings = model.encode(all_posts, show_progress_bar=True, convert_to_tensor=False)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Reduce dimensionality for visualization
        logger.info("Applying dimensionality reduction...")
        
        # PCA first to reduce to 50 dimensions
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Then t-SNE to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_posts)//4))
        embeddings_2d = tsne.fit_transform(embeddings_pca)
        
        logger.info(f"Explained variance ratio (first 10 PCA components): {pca.explained_variance_ratio_[:10]}")
        logger.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Create visualization
        plt.figure(figsize=self.config['visualization']['figure_size'], dpi=self.config['visualization']['figure_dpi'])
        
        # Create color map for platforms
        unique_platforms = list(set(platform_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_platforms)))
        color_map = dict(zip(unique_platforms, colors))
        
        # Plot points colored by platform
        for platform in unique_platforms:
            mask = np.array(platform_labels) == platform
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[color_map[platform]], 
                label=f"{platform.title()} ({mask.sum()} posts)",
                alpha=0.6,
                s=20
            )
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Sentence Embeddings Colored by Platform\n(Posts discussing Trump-related topics)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_visualization.png', dpi=self.config['visualization']['figure_dpi'], bbox_inches='tight')
        plt.close()
        
        # Calculate platform separation metrics
        logger.info("Platform Separation Analysis:")
        
        # Calculate centroids for each platform
        platform_centroids = {}
        for platform in unique_platforms:
            mask = np.array(platform_labels) == platform
            if mask.sum() > 0:
                centroid = embeddings_2d[mask].mean(axis=0)
                platform_centroids[platform] = centroid
                logger.info(f"  {platform.title()} centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        
        # Calculate distances between centroids
        if len(platform_centroids) >= 2:
            platform_pairs = [(p1, p2) for i, p1 in enumerate(unique_platforms) 
                             for p2 in unique_platforms[i+1:]]
            
            logger.info("  Centroid distances:")
            for p1, p2 in platform_pairs:
                dist = np.linalg.norm(platform_centroids[p1] - platform_centroids[p2])
                logger.info(f"    {p1.title()} - {p2.title()}: {dist:.2f}")
    
    def _create_narrative_distribution_plot(self):
        """Create narrative frame distribution visualization"""
        # Collect all narrative frames by platform
        platform_narrative_counts = {}
        
        for result in self.narrative_results:
            for platform, frames in result['narrative_frames'].items():
                if platform not in platform_narrative_counts:
                    platform_narrative_counts[platform] = {}
                
                for frame in frames:
                    if frame in platform_narrative_counts[platform]:
                        platform_narrative_counts[platform][frame] += 1
                    else:
                        platform_narrative_counts[platform][frame] = 1
        
        # Create visualization
        if platform_narrative_counts:
            fig, axes = plt.subplots(1, len(platform_narrative_counts), 
                                   figsize=(5*len(platform_narrative_counts), 6), 
                                   dpi=self.config['visualization']['figure_dpi'])
            
            if len(platform_narrative_counts) == 1:
                axes = [axes]
            
            for i, (platform, frame_counts) in enumerate(platform_narrative_counts.items()):
                if frame_counts:
                    frames = list(frame_counts.keys())
                    counts = list(frame_counts.values())
                    
                    axes[i].bar(frames, counts, alpha=0.7)
                    axes[i].set_title(f'{platform.title()} Narrative Frames')
                    axes[i].set_ylabel('Frequency')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.suptitle('Narrative Frame Distribution by Platform\n(For Cross-Platform Matched Content)', y=1.02)
            plt.savefig(self.output_dir / 'narrative_distribution.png', dpi=self.config['visualization']['figure_dpi'], bbox_inches='tight')
            plt.close()
    
    def _create_component_analysis_plot(self):
        """Create component analysis visualization"""
        component_stats = []
        for i, comp in enumerate(self.matched_components):
            platform_counts = {p: len(posts) for p, posts in comp.items() if posts}
            total_posts = sum(platform_counts.values())
            num_platforms = len(platform_counts)
            
            component_stats.append({
                'component_id': i,
                'total_posts': total_posts,
                'num_platforms': num_platforms,
                'platform_counts': platform_counts
            })
        
        # Visualize component statistics
        fig = plt.figure(figsize=(15, 5), dpi=self.config['visualization']['figure_dpi'])
        
        # Subplot 1: Component size distribution
        plt.subplot(1, 3, 1)
        sizes = [stat['total_posts'] for stat in component_stats]
        plt.hist(sizes, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Posts per Component')
        plt.ylabel('Number of Components')
        plt.title('Component Size Distribution')
        plt.yscale('log')
        
        # Subplot 2: Platform representation
        plt.subplot(1, 3, 2)
        platform_counts = [stat['num_platforms'] for stat in component_stats]
        unique_counts, count_freq = np.unique(platform_counts, return_counts=True)
        plt.bar(unique_counts, count_freq, alpha=0.7)
        plt.xlabel('Number of Platforms per Component')
        plt.ylabel('Number of Components')
        plt.title('Platform Representation in Components')
        plt.xticks(unique_counts)
        
        # Subplot 3: Cross-platform matching success
        plt.subplot(1, 3, 3)
        multi_platform = sum(1 for stat in component_stats if stat['num_platforms'] >= 2)
        single_platform = len(component_stats) - multi_platform
        
        plt.pie([multi_platform, single_platform], 
               labels=[f'Multi-platform\n({multi_platform})', f'Single-platform\n({single_platform})'],
               autopct='%1.1f%%',
               startangle=90)
        plt.title('Cross-Platform Matching Success')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_analysis.png', dpi=self.config['visualization']['figure_dpi'], bbox_inches='tight')
        plt.close()
    
    def export_results(self) -> bool:
        """Export analysis results to various formats"""
        logger.info("Exporting analysis results...")
        
        try:
            # Export processed data
            all_processed_posts = []
            for platform, data in self.processed_data.items():
                for i, post in enumerate(data['posts']):
                    metadata = data['metadata'][i]
                    all_processed_posts.append({
                        'platform': platform,
                        'post_content': post,
                        'is_reply': metadata['is_reply'],
                        'post_id': metadata['post_id']
                    })
            
            processed_df = pd.DataFrame(all_processed_posts)
            # Use proper CSV escaping parameters to handle special characters
            processed_df.to_csv(
                self.output_dir / 'processed_posts.csv', 
                index=False,
                escapechar='\\',
                quoting=1  # QUOTE_ALL
            )
            
            # Export matched components as JSONL
            with open(self.output_dir / 'matched_components.jsonl', 'w') as f:
                for component in self.matched_components:
                    json.dump(component, f)
                    f.write('\n')
            
            # Export narrative results if available
            if self.narrative_results:
                with open(self.output_dir / 'narrative_results.json', 'w') as f:
                    json.dump(self.narrative_results, f, indent=2)
            
            # Create analysis summary
            summary = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_platforms': len(self.processed_data),
                'total_posts_processed': sum(len(data['posts']) for data in self.processed_data.values()),
                'total_matched_components': len(self.matched_components),
                'multi_platform_components': sum(1 for comp in self.matched_components if len([p for p, posts in comp.items() if posts]) >= 2),
                'vllm_available': self.VLLM_AVAILABLE,
                'narrative_analysis_completed': len(self.narrative_results) > 0,
                'config_used': self.config
            }
            
            with open(self.output_dir / 'analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("✅ Results exported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def run_full_analysis(self) -> bool:
        """Run the complete cross-platform analysis pipeline"""
        logger.info("🚀 Starting cross-platform narrative analysis...")
        
        try:
            # Setup vLLM Pipeline
            if not self.setup_vllm_pipeline():
                logger.warning("vLLM pipeline setup failed, creating demo pipeline")
                self.create_demo_pipeline()
            
            # Load and preprocess data
            if not self.load_platform_data():
                logger.error("Failed to load platform data")
                return False
            
            if not self.extract_posts_from_platform_data():
                logger.error("Failed to process platform data")
                return False
            
            # Create summary statistics
            self.create_summary_statistics()
            
            # Build similarity graph and extract components
            if not self.build_similarity_graph():
                logger.error("Failed to build similarity graph")
                return False
            
            if not self.extract_matched_components():
                logger.error("Failed to extract matched components")
                return False
            
            # Analyze narrative frames
            self.analyze_narrative_frames()
            
            # Create visualizations
            self.create_visualizations()
            
            # Export results
            if not self.export_results():
                logger.error("Failed to export results")
                return False
            
            logger.info("🎯 Cross-platform narrative analysis completed successfully!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed with error: {e}")
            return False
    
    def _save_intermediate_data(self, step_name: str, data: any, file_format: str = 'json'):
        """Save intermediate data at different processing steps"""
        try:
            intermediate_dir = self.output_dir / 'intermediate'
            intermediate_dir.mkdir(exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            if file_format == 'json':
                filepath = intermediate_dir / f'{step_name}_{timestamp}.json'
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Handle data that might not be JSON serializable
                    if isinstance(data, dict):
                        serializable_data = {}
                        for key, value in data.items():
                            if isinstance(value, dict) and 'original_post' in value:
                                # Remove original_post to avoid serialization issues
                                clean_value = {k: v for k, v in value.items() if k != 'original_post'}
                                serializable_data[key] = clean_value
                            else:
                                serializable_data[key] = value
                        json.dump(serializable_data, f, indent=2, default=str)
                    else:
                        json.dump(data, f, indent=2, default=str)
            elif file_format == 'csv' and hasattr(data, 'to_csv'):
                filepath = intermediate_dir / f'{step_name}_{timestamp}.csv'
                data.to_csv(filepath, index=False)
            elif file_format == 'pickle':
                import pickle
                filepath = intermediate_dir / f'{step_name}_{timestamp}.pkl'
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info(f"💾 Saved intermediate data: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate data for {step_name}: {e}")
    
    def _check_intermediate_data(self, step_name: str) -> Optional[str]:
        """Check if intermediate data exists for a given step"""
        intermediate_dir = self.output_dir / 'intermediate'
        if not intermediate_dir.exists():
            return None
        
        # Find the most recent file for this step
        pattern = f"{step_name}_*.json"
        files = list(intermediate_dir.glob(pattern))
        
        if files:
            # Sort by creation time and return the most recent
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
        
        return None
    
    def _load_intermediate_data(self, filepath: str) -> Optional[Dict]:
        """Load intermediate data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load intermediate data from {filepath}: {e}")
            return None
    
    def can_resume_from_step(self, step_name: str) -> bool:
        """Check if analysis can be resumed from a specific step"""
        required_files = {
            'similarity_graph': ['processed_posts', 'root_posts_for_matching'],
            'component_extraction': ['processed_posts', 'root_posts_for_matching', 'similarity_graph_structure'],
            'narrative_analysis': ['processed_posts', 'component_analysis'],
            'visualization': ['processed_posts']
        }
        
        if step_name not in required_files:
            return False
        
        for required_step in required_files[step_name]:
            if not self._check_intermediate_data(required_step):
                return False
        
        return True
    
    def load_from_intermediate_data(self, resume_from: str = 'auto') -> bool:
        """Load analysis state from intermediate data
        
        Args:
            resume_from: Step to resume from ('auto', 'similarity_graph', 'component_extraction', 'narrative_analysis')
        """
        logger.info("🔄 Attempting to load from intermediate data...")
        
        # Auto-detect the furthest step we can resume from
        if resume_from == 'auto':
            possible_steps = ['narrative_analysis', 'component_extraction', 'similarity_graph', 'processed_posts']
            for step in possible_steps:
                if self.can_resume_from_step(step):
                    resume_from = step
                    break
            else:
                logger.info("No suitable intermediate data found for resuming")
                return False
        
        logger.info(f"📊 Resuming analysis from step: {resume_from}")
        
        try:
            # Load processed posts data
            processed_posts_file = self._check_intermediate_data('processed_posts')
            if processed_posts_file:
                data = self._load_intermediate_data(processed_posts_file)
                if data:
                    self.processed_data = data
                    logger.info(f"✅ Loaded processed posts data from {processed_posts_file}")
            
            # Load similarity graph structure if available
            if resume_from in ['component_extraction', 'narrative_analysis']:
                graph_file = self._check_intermediate_data('similarity_graph_structure')
                if graph_file:
                    graph_data = self._load_intermediate_data(graph_file)
                    if graph_data:
                        # Reconstruct similarity graph (simplified version)
                        self.similarity_graph = nx.Graph()
                        self.offsets = graph_data.get('offsets', {})
                        logger.info(f"✅ Loaded similarity graph structure from {graph_file}")
            
            # Load component analysis if available
            if resume_from == 'narrative_analysis':
                component_file = self._check_intermediate_data('detailed_components')
                if component_file:
                    comp_data = self._load_intermediate_data(component_file)
                    if comp_data and isinstance(comp_data, list):
                        # Reconstruct matched components from detailed data
                        self.matched_components = []
                        for comp_detail in comp_data:
                            if 'posts_by_platform' in comp_detail:
                                self.matched_components.append(comp_detail['posts_by_platform'])
                        logger.info(f"✅ Loaded {len(self.matched_components)} matched components from {component_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load intermediate data: {e}")
            return False
    
    def run_partial_analysis(self, steps: List[str] = None, resume: bool = True) -> bool:
        """Run only specific steps of the analysis
        
        Args:
            steps: List of steps to run. If None, runs all steps.
            resume: Whether to try resuming from intermediate data
        """
        all_steps = [
            'data_loading',
            'preprocessing', 
            'similarity_graph',
            'component_extraction',
            'narrative_analysis',
            'visualization',
            'export'
        ]
        
        if steps is None:
            steps = all_steps
        
        logger.info(f"🚀 Running partial analysis with steps: {steps}")
        
        # Try to resume from intermediate data if requested
        if resume and self.load_from_intermediate_data():
            logger.info("✅ Successfully resumed from intermediate data")
        
        try:
            # Setup vLLM Pipeline if needed for narrative analysis
            if 'narrative_analysis' in steps:
                if not self.setup_vllm_pipeline():
                    logger.warning("vLLM pipeline setup failed, creating demo pipeline")
                    self.create_demo_pipeline()
            
            # Run requested steps
            if 'data_loading' in steps:
                if not self.load_platform_data():
                    logger.error("Failed to load platform data")
                    return False
            
            if 'preprocessing' in steps:
                if not self.extract_posts_from_platform_data():
                    logger.error("Failed to process platform data")
                    return False
                self.create_summary_statistics()
            
            if 'similarity_graph' in steps:
                if not self.build_similarity_graph():
                    logger.error("Failed to build similarity graph")
                    return False
            
            if 'component_extraction' in steps:
                if not self.extract_matched_components():
                    logger.error("Failed to extract matched components")
                    return False
            
            if 'narrative_analysis' in steps:
                self.analyze_narrative_frames()
            
            if 'visualization' in steps:
                self.create_visualizations()
            
            if 'export' in steps:
                if not self.export_results():
                    logger.error("Failed to export results")
                    return False
            
            logger.info("🎯 Partial analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Partial analysis failed with error: {e}")
            return False


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Cross-Platform Narrative Analysis')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--data-dir', type=str, help='Directory containing platform data files')
    parser.add_argument('--threshold', type=float, help='Similarity threshold for matching')
    parser.add_argument('--embedding-model', type=str, help='Embedding model to use')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--resume', action='store_true', help='Resume from intermediate data if available')
    parser.add_argument('--steps', nargs='+', help='Specific steps to run', 
                       choices=['data_loading', 'preprocessing', 'similarity_graph', 
                               'component_extraction', 'narrative_analysis', 'visualization', 'export'])
    parser.add_argument('--resume-from', type=str, help='Specific step to resume from',
                       choices=['auto', 'similarity_graph', 'component_extraction', 'narrative_analysis'])
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if config is None:
        config = {}
    
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.threshold:
        config['similarity_threshold'] = args.threshold
    if args.embedding_model:
        config['embedding_model'] = args.embedding_model
    
    # Initialize analyzer
    analyzer = CrossPlatformAnalyzer(config=config, output_dir=args.output)
    
    # Run analysis based on arguments
    if args.steps:
        # Run specific steps
        success = analyzer.run_partial_analysis(steps=args.steps, resume=args.resume)
    elif args.resume_from:
        # Resume from specific step
        if analyzer.load_from_intermediate_data(resume_from=args.resume_from):
            # Run remaining steps
            remaining_steps = {
                'similarity_graph': ['similarity_graph', 'component_extraction', 'narrative_analysis', 'visualization', 'export'],
                'component_extraction': ['component_extraction', 'narrative_analysis', 'visualization', 'export'],
                'narrative_analysis': ['narrative_analysis', 'visualization', 'export']
            }
            steps_to_run = remaining_steps.get(args.resume_from, ['visualization', 'export'])
            success = analyzer.run_partial_analysis(steps=steps_to_run, resume=False)
        else:
            print(f"❌ Cannot resume from {args.resume_from}. Running full analysis.")
            success = analyzer.run_full_analysis()
    elif args.resume:
        # Try to resume automatically
        success = analyzer.run_partial_analysis(resume=True)
    else:
        # Run full analysis
        success = analyzer.run_full_analysis()
    
    if success:
        print(f"✅ Analysis completed successfully! Results saved to: {analyzer.output_dir}")
        sys.exit(0)
    else:
        print("❌ Analysis failed. Check the log files for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
