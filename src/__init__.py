"""Platform Narratives Analysis package.

This package provides tools for analyzing social media platform narratives
using text analysis and graph-based methods.
"""

# Core analysis modules
from .cross_platform_analyzer import CrossPlatformAnalyzer
from .reply_analyzer import ReplyAnalyzer
from .config_manager import ConfigManager

# Utility modules
from .ranking import fastLexRank
from .graph_analysis import build_graph, get_descendants, get_tree_nodes, get_posts_from_tree
from .text_processing import clean_text, detect_post_language, filter_posts_by_language
from .utils import load_json_data, create_id_to_post_map, posts_to_dataframe, extract_anchor_ids
from .similarity_graph import SimilarityGraphBuilder

# Optional vLLM wrapper (if available)
try:
    from .vllm_wrapper import VLLMWrapper, create_vllm_pipeline, check_vllm_availability
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Optional narrative classification (if dependencies available)
try:
    from .narrative_classification import NarrativeClassifier
    NARRATIVE_CLASSIFICATION_AVAILABLE = True
except ImportError:
    NARRATIVE_CLASSIFICATION_AVAILABLE = False

__all__ = [
    # Core analyzers
    'CrossPlatformAnalyzer',
    'ReplyAnalyzer',
    'ConfigManager',
    
    # Utility functions
    'fastLexRank',
    'build_graph',
    'get_descendants',
    'get_tree_nodes',
    'get_posts_from_tree',
    'clean_text',
    'detect_post_language',
    'filter_posts_by_language',
    'load_json_data',
    'create_id_to_post_map',
    'posts_to_dataframe',
    'extract_anchor_ids',
    'SimilarityGraphBuilder',
    
    # Optional components
    'VLLM_AVAILABLE',
    'NARRATIVE_CLASSIFICATION_AVAILABLE'
]

# Add optional components to __all__ if available
if VLLM_AVAILABLE:
    __all__.extend(['VLLMWrapper', 'create_vllm_pipeline', 'check_vllm_availability'])

if NARRATIVE_CLASSIFICATION_AVAILABLE:
    __all__.extend(['NarrativeClassifier'])