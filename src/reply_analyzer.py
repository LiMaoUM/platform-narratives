"""
Reply Analyzer Module

Refactored from reply_analysis_complete.py to be a proper module.
Handles reply classification and conversation graph analysis.
"""

import os
import sys
import json
import logging
import warnings
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import re

# Import internal modules
from .vllm_wrapper import create_vllm_pipeline, check_vllm_availability

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ReplyAnalyzer:
    """
    Analyzer for studying replies to root posts across social media platforms with vLLM classification.
    """
    
    def __init__(self, output_dir: str = "reply_analysis_output"):
        """
        Initialize the reply analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define reply analysis categories (3 types as specified)
        self.reply_types = {
            'reinforce': 'Reinforce - Support original post stance and narrative',
            'challenge': 'Challenge - Refute or oppose',
            'shift': 'Shift - Introduce new perspectives or go off-topic'
        }
        
        # Define platforms
        self.platforms = ['truth', 'bluesky', 'mastodon']
        
        # Initialize vLLM pipeline
        self.vllm_pipeline = None
        self.vllm_available = False
        
        logger.info(f"Initialized ReplyAnalyzer")
        logger.info(f"Output directory: {self.output_dir}")
    
    def clean_text(self, content: str, platform: str) -> str:
        """Clean text by removing HTML, mentions, hashtags, and URLs.
        
        Args:
            content: String text to clean
            platform: Platform name for platform-specific cleaning
            
        Returns:
            Cleaned text string
        """
        if not isinstance(content, str):
            return ""
            
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
    
    def setup_vllm_pipeline(self) -> bool:
        """Setup vLLM Pipeline for reply classification"""
        logger.info("Setting up vLLM Pipeline for reply classification...")
        
        # Respect the caller's GPU selection; set CUDA_VISIBLE_DEVICES in the
        # environment to pin specific GPUs.
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            logger.info(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        try:
            # Check if vLLM is available
            if not check_vllm_availability():
                logger.error("❌ vLLM not available")
                return False
            
            # Create vLLM pipeline with higher GPU memory utilization to avoid CUDA memory issues
            logger.info("Initializing vLLM pipeline with optimized memory settings...")
            self.vllm_pipeline = create_vllm_pipeline(
                model_name="google/gemma-3-27b-it",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95
            )
            
            logger.info("✅ vLLM loaded successfully!")
            self.vllm_available = True
            
            # Test the pipeline
            logger.info("Testing reply classification with vLLM pipeline...")
            test_root = "Trump's policies will make America great again!"
            test_reply = "I completely agree, his economic plan is brilliant."
            
            test_result = self.classify_reply_relationship(test_root, test_reply)
            logger.info(f"Test classification result: {test_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup vLLM pipeline: {e}")
            self.vllm_available = False
            return False
    
    def classify_reply_relationship(self, root_post: str, reply_post: str) -> Dict[str, Any]:
        """
        Classify the relationship between a root post and its reply using vLLM.
        
        Args:
            root_post: The original post content
            reply_post: The reply content
            
        Returns:
            Dictionary with classification results
        """
        if not self.vllm_available or self.vllm_pipeline is None:
            return {
                'classification': 'shift',
                'confidence': 0.0,
                'reasoning': 'vLLM not available',
                'error': 'vLLM pipeline not initialized'
            }
        
        # Create classification prompt
        prompt = f"""Analyze the relationship between this root post and reply:

ROOT POST: "{root_post}"

REPLY: "{reply_post}"

Classify the reply's stance toward the root post into one of these THREE categories:
- REINFORCE: Support the original post's stance and narrative
- CHALLENGE: Challenge or oppose the original post
- SHIFT: Introduce new perspectives or go off-topic

Respond in JSON format:
{{
    "classification": "REINFORCE|CHALLENGE|SHIFT",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of the classification"
}}"""

        try:
            # Create messages for vLLM pipeline
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Generate response using vLLM
            response = self.vllm_pipeline(
                messages,
                max_new_tokens=200,
                temperature=0.1
            )
            
            # Extract generated text
            response_text = response[0]["generated_text"][0]["content"].strip()
            
            # Parse the JSON response
            if not response_text.endswith('}') and '{' in response_text:
                json_start = response_text.find('{')
                response_text = response_text[json_start:]
            
            try:
                result = json.loads(response_text)
                return {
                    'classification': result.get('classification', 'shift').lower().replace('reinforce', 'reinforce'),
                    'confidence': float(result.get('confidence', 0.0)),
                    'reasoning': result.get('reasoning', 'No reasoning provided')
                }
                
            except json.JSONDecodeError:
                return {
                    'classification': 'shift',
                    'confidence': 0.0,
                    'reasoning': 'Failed to parse JSON response',
                    'raw_response': response_text
                }
                
        except Exception as e:
            logger.error(f"Error in reply classification: {e}")
            return {
                'classification': 'shift',
                'confidence': 0.0,
                'reasoning': f'Classification error: {str(e)}',
                'error': str(e)
            }
    
    def load_matched_components(self, matched_components_path: str) -> List[Dict]:
        """
        Load matched components data.
        
        Args:
            matched_components_path: Path to matched components JSON file
            
        Returns:
            List of matched components
        """
        logger.info("Loading matched components...")
        
        with open(matched_components_path, 'r', encoding='utf-8') as f:
            components = json.load(f)
        
        logger.info(f"Loaded {len(components)} matched components")
        
        # Filter components that have posts from multiple platforms
        multi_platform_components = []
        for comp in components:
            platforms_with_posts = [p for p in self.platforms if p in comp and comp[p]]
            if len(platforms_with_posts) >= 2:
                multi_platform_components.append(comp)
        
        logger.info(f"Found {len(multi_platform_components)} components with multiple platforms")
        return multi_platform_components
    
    def load_conversation_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load conversation data from platform files to identify reply relationships.
        
        Args:
            data_dir: Directory containing platform data files
            
        Returns:
            Dictionary mapping platform names to DataFrames with conversation data
        """
        logger.info("Loading conversation data from platform files...")
        
        platform_files = {
            'truth': 'truthsocial.trump.json',
            'bluesky': 'bsky.trump.json', 
            'mastodon': 'mastodon.trump.json'
        }
        
        conversation_data = {}
        
        for platform, filename in platform_files.items():
            file_path = Path(data_dir) / filename
            
            if file_path.exists():
                logger.info(f"Loading {platform} conversation data from {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Process based on platform structure
                if platform == 'truth':
                    processed_df = self._process_truth_data(raw_data)
                elif platform == 'bluesky':
                    processed_df = self._process_bluesky_data(raw_data)
                elif platform == 'mastodon':
                    processed_df = self._process_mastodon_data(raw_data)
                
                conversation_data[platform] = processed_df
                logger.info(f"  - Loaded {len(processed_df)} posts from {platform}")
            else:
                logger.warning(f"File not found: {file_path}")
                conversation_data[platform] = pd.DataFrame()
        
        return conversation_data
    
    def _process_truth_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process Truth Social data structure."""
        processed_data = []
        
        for post in data:
            processed_data.append({
                'id': post.get('id', ''),
                'content': post.get('content', ''),
                'in_reply_to_id': post.get('in_reply_to_id'),
                'created_at': post.get('created_at'),
                'platform': 'truth'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_bluesky_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process Bluesky data structure."""
        processed_data = []
        
        for post in data:
            # Handle nested record structure for Bluesky
            record = post.get('record', {})
            processed_data.append({
                'id': post.get('uri', ''),
                'content': record.get('text', ''),
                'in_reply_to_id': record.get('reply', {}).get('parent', {}).get('uri') if record.get('reply') else None,
                'created_at': record.get('createdAt'),
                'platform': 'bluesky'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_mastodon_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process Mastodon data structure."""
        processed_data = []
        
        for post in data:
            processed_data.append({
                'id': post.get('id', ''),
                'content': post.get('content', ''),
                'in_reply_to_id': post.get('in_reply_to_id'),
                'created_at': post.get('created_at'),
                'platform': 'mastodon'
            })
        
        return pd.DataFrame(processed_data)
    
    def build_conversation_graphs(self, conversation_data: Dict[str, pd.DataFrame]) -> Dict[str, nx.DiGraph]:
        """
        Build conversation graphs for each platform.
        
        Args:
            conversation_data: Dictionary of platform DataFrames
            
        Returns:
            Dictionary mapping platform names to conversation graphs
        """
        logger.info("Building conversation graphs...")
        
        graphs = {}
        
        for platform, df in conversation_data.items():
            logger.info(f"Building graph for {platform}...")
            
            G = nx.DiGraph()
            
            # Add all posts as nodes
            for _, row in df.iterrows():
                G.add_node(row['id'], 
                          content=row['content'], 
                          created_at=row['created_at'],
                          platform=platform)
            
            # Add reply edges
            reply_count = 0
            for _, row in df.iterrows():
                if pd.notna(row['in_reply_to_id']) and row['in_reply_to_id'] in G.nodes():
                    G.add_edge(row['in_reply_to_id'], row['id'])
                    reply_count += 1
            
            graphs[platform] = G
            logger.info(f"  Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} reply edges")
            logger.info(f"  Root posts: {len([n for n in G.nodes() if G.in_degree(n) == 0])}")
        
        return graphs
    
    def identify_root_posts_in_components(self, 
                                        matched_components: List[Dict],
                                        conversation_graphs: Dict[str, nx.DiGraph]) -> List[Dict]:
        """
        Identify root posts within matched components using cleaned content mapping.
        
        Args:
            matched_components: List of matched components
            conversation_graphs: Platform conversation graphs
            
        Returns:
            List of components with identified root posts and their replies
        """
        logger.info("Identifying root posts in matched components...")
        logger.info(f"Processing {len(matched_components)} components...")
        
        # Pre-build content-to-node mappings for each platform using cleaned content
        logger.info("Building content-to-node mappings with cleaned text for faster lookup...")
        content_mappings = {}
        for platform, graph in conversation_graphs.items():
            logger.info(f"Mapping content for {platform} ({graph.number_of_nodes()} nodes)...")
            content_mappings[platform] = {}
            for node_id in graph.nodes():
                original_content = graph.nodes[node_id]['content']
                cleaned_content = self.clean_text(original_content, platform)
                content_mappings[platform][cleaned_content] = node_id
        
        logger.info("Content mappings built. Processing components...")
        components_with_replies = []
        
        for comp_idx, component in enumerate(matched_components):
            if comp_idx % 100 == 0:
                logger.info(f"Processing component {comp_idx}/{len(matched_components)}")
            
            component_analysis = {
                'component_id': comp_idx,
                'platforms': {}
            }
            
            for platform in self.platforms:
                if platform not in component or not component[platform]:
                    continue
                
                platform_data = {
                    'root_posts': []
                }
                
                # Find root posts for this platform in the component
                for post_data in component[platform]:
                    post_content = post_data.get('content', '')
                    cleaned_content = self.clean_text(post_content, platform)
                    
                    # Find corresponding node in conversation graph
                    if cleaned_content in content_mappings[platform]:
                        node_id = content_mappings[platform][cleaned_content]
                        
                        # Check if this is a root post (no incoming edges)
                        if conversation_graphs[platform].in_degree(node_id) == 0:
                            # Get replies to this root post
                            replies = self._get_nested_replies(conversation_graphs[platform], node_id, 1)
                            
                            root_post_data = {
                                'id': node_id,
                                'content': post_content,
                                'cleaned_content': cleaned_content,
                                'replies': replies,
                                'reply_count': len(replies)
                            }
                            
                            platform_data['root_posts'].append(root_post_data)
                
                if platform_data['root_posts']:
                    component_analysis['platforms'][platform] = platform_data
            
            # Only include components that have conversation data
            if component_analysis['platforms']:
                components_with_replies.append(component_analysis)
        
        logger.info(f"Found {len(components_with_replies)} components with conversation data")
        return components_with_replies
    
    def _get_nested_replies(self, graph: nx.DiGraph, parent_id: str, depth: int, max_depth: int = 5) -> List[Dict]:
        """
        Recursively get nested replies up to max_depth.
        
        Args:
            graph: Conversation graph
            parent_id: ID of parent post
            depth: Current depth
            max_depth: Maximum depth to traverse
            
        Returns:
            List of nested reply data
        """
        if depth > max_depth:
            return []
        
        nested_replies = []
        for reply_id in graph.successors(parent_id):
            reply_data = {
                'id': reply_id,
                'content': graph.nodes[reply_id]['content'],
                'created_at': graph.nodes[reply_id]['created_at'],
                'depth': depth
            }
            
            # Recursively get deeper replies
            reply_data['nested_replies'] = self._get_nested_replies(graph, reply_id, depth + 1, max_depth)
            nested_replies.append(reply_data)
        
        return nested_replies
    
    def classify_replies_in_components(self, components_with_replies: List[Dict]) -> List[Dict]:
        """
        Classify all replies in the components using vLLM batch processing.
        
        Args:
            components_with_replies: Components with conversation analysis
            
        Returns:
            Components with reply classifications added
        """
        if not self.vllm_available:
            logger.warning("⚠️ vLLM not available, skipping reply classification")
            return components_with_replies
        
        logger.info("Classifying replies using vLLM batch processing...")
        
        # Deep copy the components to avoid modifying the original
        classified_components = copy.deepcopy(components_with_replies)
        
        total_replies_processed = 0
        batch_size = 16  # Adjust based on model and memory constraints
        
        for component in tqdm(classified_components, desc="Processing components"):
            for platform in component['platforms']:
                for root_post in component['platforms'][platform]['root_posts']:
                    root_content = root_post['content']
                    
                    # Classify direct replies
                    for reply in root_post['replies']:
                        reply_content = reply['content']
                        
                        classification_result = self.classify_reply_relationship(
                            root_content, reply_content
                        )
                        
                        reply['classification'] = classification_result
                        total_replies_processed += 1
                        
                        # Classify nested replies recursively
                        if 'nested_replies' in reply and reply['nested_replies']:
                            total_replies_processed += self._classify_nested_replies(
                                root_content, reply['nested_replies']
                            )
        
        logger.info(f"Classified {total_replies_processed} replies across {len(classified_components)} components")
        return classified_components
    
    def _classify_nested_replies(self, root_content: str, nested_replies: List[Dict]) -> int:
        """
        Recursively classify nested replies.
        
        Args:
            root_content: The original root post content
            nested_replies: List of nested reply dictionaries
            
        Returns:
            Number of replies processed
        """
        replies_processed = 0
        
        for nested_reply in nested_replies:
            reply_content = nested_reply['content']
            
            classification_result = self.classify_reply_relationship(
                root_content, reply_content
            )
            
            nested_reply['classification'] = classification_result
            replies_processed += 1
            
            # Recursively classify deeper nested replies
            if 'nested_replies' in nested_reply and nested_reply['nested_replies']:
                replies_processed += self._classify_nested_replies(
                    root_content, nested_reply['nested_replies']
                )
        
        return replies_processed
