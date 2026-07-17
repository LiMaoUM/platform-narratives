#!/usr/bin/env python3
"""
Reply Analysis for Root Posts - Platform Narratives Analysis

This script analyzes replies to root posts identified in matched components,
building on the existing narrative divergence analysis framework.
Includes vLLM-based reply classification.

Author: Data Analyst
Date: June 6, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os
import warnings
import networkx as nx
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import re

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))
from vllm_wrapper import VLLMWrapper, create_vllm_pipeline, check_vllm_availability

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [STEP] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class ReplyAnalyzerWithVLLM:
    """
    Analyzer for studying replies to root posts across social media platforms with vLLM classification.
    """
    
    def __init__(self, output_dir: str = "narrative_analysis_output"):
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
        
        logger.info(f"Initialized ReplyAnalyzerWithVLLM")
        logger.info(f"Output directory: {self.output_dir}")
    
    def clean_text(self, content: str, platform: str) -> str:
        """Clean text by removing HTML, mentions, hashtags, and URLs.
        
        Args:
            text: String text to clean
            
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
        
        # Set CUDA devices to use GPU 1 only
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info("🔧 Set CUDA_VISIBLE_DEVICES to '1'")
        
        try:
            # Check if vLLM is available
            if not check_vllm_availability():
                logger.error("❌ vLLM not available")
                logger.info("💡 Install vLLM: pip install vllm")
                return False
            
            # Create vLLM pipeline with higher GPU memory utilization to avoid CUDA memory issues
            logger.info("Initializing vLLM pipeline with optimized memory settings...")
            self.vllm_pipeline = create_vllm_pipeline(
                model_name="google/gemma-3-27b-it",  # Use original Gemma 3 27B model
                tensor_parallel_size=1,  # Use single GPU first
                gpu_memory_utilization=0.95  # Use 95% of GPU memory for better utilization
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
                # Try to extract JSON from response
                json_start = response_text.find('{')
                json_part = response_text[json_start:]
                if not json_part.endswith('}'):
                    json_part += '}'
                response_text = json_part
            
            try:
                result = json.loads(response_text)
                
                # Normalize classification to lowercase
                classification = result.get('classification', 'shift').lower()
                if classification not in ['reinforce', 'challenge', 'shift']:
                    classification = 'shift'
                
                return {
                    'classification': classification,
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', 'No reasoning provided')
                }
                
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                response_lower = response_text.lower()
                
                if 'reinforce' in response_lower:
                    classification = 'reinforce'
                elif 'challenge' in response_lower:
                    classification = 'challenge'
                else:
                    classification = 'shift'
                
                return {
                    'classification': classification,
                    'confidence': 0.5,
                    'reasoning': 'Parsed from non-JSON response',
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
                logger.info(f"Loading {platform} data from {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to DataFrame based on platform structure
                if platform == 'truth':
                    df = self._process_truth_data(data)
                elif platform == 'bluesky':
                    df = self._process_bluesky_data(data)
                elif platform == 'mastodon':
                    df = self._process_mastodon_data(data)
                
                conversation_data[platform] = df
                logger.info(f"  Loaded {len(df)} posts for {platform}")
            else:
                logger.warning(f"Warning: {file_path} not found")
        
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
                if pd.notna(row['in_reply_to_id']) and row['in_reply_to_id'] in G.nodes:
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
        Optimized: Identify root posts within matched components using cleaned content mapping.
        
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
                # Use both original content and cleaned content for better matching
                original_content = graph.nodes[node_id]['content']
                cleaned_content = self.clean_text(original_content, platform=platform)
                if "Fauci is on the run" in cleaned_content:
                    logger.warning(f"Found sensitive content in {platform} node {node_id}: {cleaned_content}")
                # Map both versions for more robust matching
                
                if cleaned_content and cleaned_content not in content_mappings[platform]:
                    content_mappings[platform][cleaned_content] = []
                if cleaned_content:
                    content_mappings[platform][cleaned_content].append(node_id)

        # save content_mappings to file for debugging
        with open(self.output_dir / 'content_mappings.json', 'w', encoding='utf-8') as f:
            json.dump(content_mappings, f, indent=2)
        
        logger.info("Content mappings built. Processing components...")
        components_with_replies = []
        
        for comp_idx, component in enumerate(matched_components):
            if comp_idx % 100 == 0:  # Progress update
                logger.info(f"Processing component {comp_idx}/{len(matched_components)}")
            
            component_analysis = {
                'component_id': comp_idx,
                'platforms': {}
            }
            
            for platform in self.platforms:
                if platform not in component or not component[platform]:
                    continue
                
                platform_posts = component[platform]
                graph = conversation_graphs.get(platform)
                content_mapping = content_mappings.get(platform)
                
                if graph is None or content_mapping is None:
                    continue
                
                root_posts = []
                
                for post_content in platform_posts:
                    cleaned_post_content = self.clean_text(post_content, platform=platform)
                    
                    # Try lookup with both original and cleaned content
                    found = False
                    matching_node_ids = []
                    for content_to_check in [post_content, cleaned_post_content]:
                        if content_to_check and content_to_check in content_mapping:
                            matching_node_ids.extend(content_mapping[content_to_check])
                            found = True
                    
                    if not found:
                        continue
                    
                    # Remove duplicates while preserving order
                    matching_node_ids = list(dict.fromkeys(matching_node_ids))
                    
                    for post_id in matching_node_ids:
                        # Get direct replies (successors)
                        replies = []
                        for successor_id in graph.successors(post_id):
                            reply_data = {
                                'id': successor_id,
                                'content': graph.nodes[successor_id]['content'],
                                'created_at': graph.nodes[successor_id]['created_at']
                            }
                            replies.append(reply_data)
                        
                        # Only include if it has replies
                        if replies:
                            root_post_data = {
                                'id': post_id,
                                'content': graph.nodes[post_id]['content'],
                                'created_at': graph.nodes[post_id]['created_at'],
                                'replies': replies
                            }
                            root_posts.append(root_post_data)
                
                if root_posts:
                    component_analysis['platforms'][platform] = {
                        'root_posts': root_posts,
                        'total_posts': len(platform_posts)
                    }
            
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
        import copy
        classified_components = copy.deepcopy(components_with_replies)
        
        total_replies_processed = 0
        batch_size = 16  # Adjust based on model and memory constraints
        
        for component in tqdm(classified_components, desc="Processing components"):
            for platform in component['platforms']:
                platform_data = component['platforms'][platform]
                
                for root_post in platform_data['root_posts']:
                    root_content = root_post['content']
                    
                    # Collect all direct replies for this root post
                    direct_replies = root_post['replies']
                    if not direct_replies:
                        continue
                    
                    # Process all direct replies in batches
                    for i in range(0, len(direct_replies), batch_size):
                        batch_pairs = []
                        batch_indices = []
                        
                        # Collect batch pairs and their indices
                        for j in range(i, min(i + batch_size, len(direct_replies))):
                            reply = direct_replies[j]
                            batch_pairs.append((root_content, reply['content']))
                            batch_indices.append(j)
                        
                        # Process this batch
                        batch_results = self._batch_classify_replies(batch_pairs)
                        
                        # Map results back to replies
                        for idx, result_idx in enumerate(batch_indices):
                            if idx < len(batch_results):
                                direct_replies[result_idx]['classification'] = batch_results[idx]
                        
                        total_replies_processed += len(batch_pairs)
                    
                    # Process nested replies (if any) with the same batching approach
                    for reply in direct_replies:
                        if 'nested_replies' in reply and reply['nested_replies']:
                            nested_count = self._batch_classify_nested_replies(
                                root_content, 
                                reply['nested_replies'],
                                batch_size
                            )
                            total_replies_processed += nested_count
            
            # classified_components.append(classified_component)
        
        logger.info(f"Classified {total_replies_processed} replies across {len(classified_components)} components using batch processing")
        return classified_components
    
    def _classify_nested_replies(self, root_content: str, nested_replies: List[Dict]) -> None:
        """
        Recursively classify nested replies.
        
        Args:
            root_content: The original root post content
            nested_replies: List of nested reply dictionaries
        """
        for nested_reply in nested_replies:
            reply_content = nested_reply['content']
            
            classification_result = self.classify_reply_relationship(
                root_content, reply_content
            )
            
            nested_reply['classification'] = classification_result
            
            # Recursively classify deeper nested replies
            if 'nested_replies' in nested_reply and nested_reply['nested_replies']:
                self._classify_nested_replies(root_content, nested_reply['nested_replies'])
    
    def analyze_reply_classifications(self, classified_components: List[Dict]) -> pd.DataFrame:
        """
        Analyze the distribution of reply classifications across platforms.
        
        Args:
            classified_components: Components with classified replies
            
        Returns:
            DataFrame with classification analysis
        """
        logger.info("Analyzing reply classifications...")
        
        analysis_results = []
        
        for component in tqdm(classified_components, desc="Analyzing classifications"):
            component_id = component['component_id']
            
            # Collect classification data for each platform
            platform_classifications = {}
            
            for platform, data in component['platforms'].items():
                classifications = defaultdict(int)
                total_replies = 0
                confidence_scores = []
                
                for root_post in data['root_posts']:
                    # Count direct replies
                    for reply in root_post['replies']:
                        if 'classification' in reply:
                            classification = reply['classification']['classification']
                            confidence = reply['classification']['confidence']
                            
                            classifications[classification] += 1
                            confidence_scores.append(confidence)
                            total_replies += 1
                        
                        # Count nested replies
                        nested_count, nested_classifications, nested_confidences = self._count_nested_classifications(
                            reply.get('nested_replies', [])
                        )
                        
                        for class_type, count in nested_classifications.items():
                            classifications[class_type] += count
                        
                        confidence_scores.extend(nested_confidences)
                        total_replies += nested_count
                
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                
                platform_classifications[platform] = {
                    'total_classified_replies': total_replies,
                    'avg_classification_confidence': avg_confidence,
                    **{f'{class_type}_count': count for class_type, count in classifications.items()},
                    **{f'{class_type}_ratio': count/total_replies if total_replies > 0 else 0 
                       for class_type, count in classifications.items()}
                }
            
            # Calculate cross-platform classification divergence
            platforms_with_data = list(platform_classifications.keys())
            
            if len(platforms_with_data) >= 2:
                # Calculate divergence for each classification type
                classification_divergences = {}
                
                for class_type in self.reply_types.keys():
                    ratios = []
                    for platform in platforms_with_data:
                        ratio_key = f'{class_type}_ratio'
                        if ratio_key in platform_classifications[platform]:
                            ratios.append(platform_classifications[platform][ratio_key])
                    
                    if len(ratios) >= 2:
                        divergence = np.std(ratios) / (np.mean(ratios) + 1e-8)
                        classification_divergences[f'{class_type}_divergence'] = divergence
                
                analysis_results.append({
                    'component_id': component_id,
                    'platforms_analyzed': platforms_with_data,
                    'platform_count': len(platforms_with_data),
                    **{f'{p}_{metric}': value 
                       for p, metrics in platform_classifications.items() 
                       for metric, value in metrics.items()},
                    **classification_divergences,
                    'total_cross_platform_classified_replies': sum(
                        platform_classifications[p]['total_classified_replies'] 
                        for p in platforms_with_data
                    )
                })
        
        results_df = pd.DataFrame(analysis_results)
        logger.info(f"Analyzed classifications for {len(results_df)} components")
        
        return results_df
    
    def _count_nested_classifications(self, nested_replies: List[Dict]) -> Tuple[int, Dict[str, int], List[float]]:
        """
        Count classifications in nested replies recursively.
        
        Args:
            nested_replies: List of nested reply dictionaries
            
        Returns:
            Tuple of (total_count, classification_counts, confidence_scores)
        """
        total_count = 0
        classifications = defaultdict(int)
        confidence_scores = []
        
        for nested_reply in nested_replies:
            if 'classification' in nested_reply:
                classification = nested_reply['classification']['classification']
                confidence = nested_reply['classification']['confidence']
                
                classifications[classification] += 1
                confidence_scores.append(confidence)
                total_count += 1
            
            # Recursively count deeper nested replies
            if 'nested_replies' in nested_reply and nested_reply['nested_replies']:
                nested_count, nested_classifications, nested_confidences = self._count_nested_classifications(
                    nested_reply['nested_replies']
                )
                
                total_count += nested_count
                confidence_scores.extend(nested_confidences)
                
                for class_type, count in nested_classifications.items():
                    classifications[class_type] += count
        
        return total_count, dict(classifications), confidence_scores
    
    def _batch_classify_replies(self, reply_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of (root_post, reply_post) pairs using vLLM.
        
        Args:
            reply_pairs: List of (root_post, reply_post) tuples
            
        Returns:
            List of classification results (dictionaries)
        """
        if not self.vllm_available or self.vllm_pipeline is None:
            # Return default classifications if vLLM is not available
            return [
                {
                    'classification': 'shift',
                    'confidence': 0.0,
                    'reasoning': 'vLLM not available',
                    'error': 'vLLM pipeline not initialized'
                }
                for _ in reply_pairs
            ]
        
        # Clean texts and prepare prompts
        batch_messages = []
        
        for root_post, reply_post in reply_pairs:
            # Clean both texts (assuming we don't know the platform, use a generic approach)
            root_clean = self.clean_text(root_post, platform='truth')  # Use truth as default for cleaning
            reply_clean = self.clean_text(reply_post, platform='truth')  # Use truth as default for cleaning
            
            # Create classification prompt
            prompt = f"""Analyze the relationship between this root post and reply:

ROOT POST: "{root_clean}"

REPLY: "{reply_clean}"

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
            
            # Create messages for vLLM pipeline
            batch_messages.append([{"role": "user", "content": prompt}])
        
        try:
            # Generate responses using vLLM in batch
            batch_responses = self.vllm_pipeline(
                batch_messages,
                max_new_tokens=200,
                temperature=0.1
            )
            
            # Parse all responses
            results = []
            
            for i, response in enumerate(batch_responses):
                try:
                    response_text = response["generated_text"][0]["content"].strip()
                    
                    # Extract JSON from response if not formatted correctly
                    if not response_text.endswith('}') and '{' in response_text:
                        json_start = response_text.find('{')
                        json_part = response_text[json_start:]
                        if not json_part.endswith('}'):
                            json_part += '}'
                        response_text = json_part
                    
                    try:
                        result = json.loads(response_text)
                        
                        # Normalize classification to lowercase
                        classification = result.get('classification', 'shift').lower()
                        if classification not in ['reinforce', 'challenge', 'shift']:
                            classification = 'shift'
                        
                        results.append({
                            'classification': classification,
                            'confidence': float(result.get('confidence', 0.5)),
                            'reasoning': result.get('reasoning', 'No reasoning provided')
                        })
                        
                    except json.JSONDecodeError:
                        # Fallback parsing if JSON is malformed
                        response_lower = response_text.lower()
                        
                        if 'reinforce' in response_lower:
                            classification = 'reinforce'
                        elif 'challenge' in response_lower:
                            classification = 'challenge'
                        else:
                            classification = 'shift'
                        
                        results.append({
                            'classification': classification,
                            'confidence': 0.5,
                            'reasoning': 'Parsed from non-JSON response',
                            'raw_response': response_text
                        })
                        
                except Exception as response_error:
                    logger.error(f"Error parsing response {i}: {response_error}")
                    results.append({
                        'classification': 'shift',
                        'confidence': 0.0,
                        'reasoning': f'Response parsing error: {str(response_error)}',
                        'error': str(response_error)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch reply classification: {e}")
            return [
                {
                    'classification': 'shift',
                    'confidence': 0.0,
                    'reasoning': f'Batch classification error: {str(e)}',
                    'error': str(e)
                }
                for _ in reply_pairs
            ]
    
    def _batch_classify_nested_replies(
        self, 
        root_content: str, 
        nested_replies: List[Dict], 
        batch_size: int = 16
    ) -> int:
        """
        Recursively classify nested replies in batches.
        
        Args:
            root_content: The original root post content
            nested_replies: List of nested reply dictionaries
            batch_size: Size of batches for processing
            
        Returns:
            Total number of nested replies processed
        """
        if not nested_replies:
            return 0
        
        total_processed = 0
        
        # Process this level of nested replies in batches
        for i in range(0, len(nested_replies), batch_size):
            batch_end = min(i + batch_size, len(nested_replies))
            batch_pairs = []
            batch_indices = []
            
            # Collect batch pairs
            for j in range(i, batch_end):
                reply = nested_replies[j]
                batch_pairs.append((root_content, reply['content']))
                batch_indices.append(j)
            
            # Process this batch
            batch_results = self._batch_classify_replies(batch_pairs)
            
            # Map results back to replies
            for idx, result_idx in enumerate(batch_indices):
                if idx < len(batch_results):
                    nested_replies[result_idx]['classification'] = batch_results[idx]
            
            total_processed += len(batch_pairs)
        
        # Recursively process deeper nested replies
        for nested_reply in nested_replies:
            if 'nested_replies' in nested_reply and nested_reply['nested_replies']:
                total_processed += self._batch_classify_nested_replies(
                    root_content, 
                    nested_reply['nested_replies'], 
                    batch_size
                )
        
        return total_processed
        

def main():
    """Main function to run the reply analysis with vLLM."""
    logger.info("Starting Reply Analysis for Root Posts with vLLM Classification")
    logger.info("=" * 60)
    
    import time
    start_time = time.time()
    
    # Initialize analyzer
    analyzer = ReplyAnalyzerWithVLLM()
    
    # Configuration
    matched_components_path = "output/matched_components.json"
    data_dir = "data/data"
    
    # For testing: limit number of components to process
    
    try:
        # Step 1: Setup vLLM pipeline for reply classification
        logger.info("Step 1: Setting up vLLM pipeline...")
        step_start = time.time()
        vllm_setup_success = analyzer.setup_vllm_pipeline()
        logger.info(f"  ⏱️ Step 1 completed in {time.time() - step_start:.2f} seconds")
        
        # Step 2: Load matched components
        logger.info("Step 2: Loading matched components...")
        step_start = time.time()
        matched_components = analyzer.load_matched_components(matched_components_path)
        matched_components = matched_components[1000:]  # Limit for testing 
        
        logger.info(f"  ⏱️ Step 2 completed in {time.time() - step_start:.2f} seconds")
        
        # Step 3: Load conversation data
        logger.info("Step 3: Loading conversation data...")
        step_start = time.time()
        conversation_data = analyzer.load_conversation_data(data_dir)
        logger.info(f"  ⏱️ Step 3 completed in {time.time() - step_start:.2f} seconds")
        
        # Step 4: Build conversation graphs
        logger.info("Step 4: Building conversation graphs...")
        step_start = time.time()
        conversation_graphs = analyzer.build_conversation_graphs(conversation_data)
        logger.info(f"  ⏱️ Step 4 completed in {time.time() - step_start:.2f} seconds")
        
        # Step 5: Identify root posts and replies in components (OPTIMIZED)
        logger.info("Step 5: Identifying root posts and replies (OPTIMIZED)...")
        step_start = time.time()
        components_with_replies = analyzer.identify_root_posts_in_components(
            matched_components, conversation_graphs
        )
        logger.info(f"  ⏱️ Step 5 completed in {time.time() - step_start:.2f} seconds")
        
        # Step 6: Classify replies using vLLM (if available)
        logger.info("Step 6: Classifying replies...")
        step_start = time.time()
        if vllm_setup_success:
            classified_components = analyzer.classify_replies_in_components(components_with_replies)
            
            # Analyze reply classifications
            logger.info("Step 7: Analyzing reply classifications...")
            classification_analysis_df = analyzer.analyze_reply_classifications(classified_components)
            
            # Save classification results
            classification_results_path = analyzer.output_dir / 'reply_classification_analysis.csv'
            classification_analysis_df.to_csv(classification_results_path, index=False)
            logger.info(f"Classification analysis saved to: {classification_results_path}")
        else:
            logger.info("⚠️ Skipping reply classification due to vLLM setup failure")
            classified_components = components_with_replies
        logger.info(f"  ⏱️ Step 6 completed in {time.time() - step_start:.2f} seconds")
        
        # Save detailed conversation data with classifications
        conversation_data_path = analyzer.output_dir / 'components_with_conversation_and_classification_data.json'
        with open(conversation_data_path, 'w', encoding='utf-8') as f:
            json.dump(classified_components if vllm_setup_success else components_with_replies, 
                     f, indent=2, default=str)
        logger.info(f"Detailed conversation data saved to: {conversation_data_path}")
        
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("Reply Analysis with vLLM Classification Complete!")
        logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Results saved in: {analyzer.output_dir}")
        
        if vllm_setup_success:
            logger.info("✅ Reply classification completed using vLLM")
            logger.info(f"✅ Classification analysis results available")
        else:
            logger.info("ℹ️ Reply classification skipped (vLLM not available)")
        
    except Exception as e:
        logger.info(f"Error during analysis: {e}")
        import traceback
        traceback.logger.info_exc()


if __name__ == "__main__":
    main()
