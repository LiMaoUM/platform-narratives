"""
Configuration Manager

Handles loading and managing configuration files for the analysis pipeline.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if self.config_path and Path(self.config_path).exists():
            logger.info(f"Loading configuration from: {self.config_path}")
            return self._load_config_file(self.config_path)
        else:
            logger.info("Using default configuration")
            return self._default_config()
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_file = Path(config_path)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Falling back to default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'data': {
                'data_dir': 'data/data',
                'platforms': {
                    'truthsocial': 'truthsocial.trump.json',
                    'bluesky': 'bsky.trump.json',
                    'mastodon': 'mastodon.trump.json'
                },
                'max_posts_per_platform': None,
                'min_text_length': 10,
                'language_filter': 'en'
            },
            'similarity': {
                'model_name': 'all-MiniLM-L6-v2',
                'threshold': 0.65
            },
            'embedding': {
                'sample_size': 2000
            },
            'narrative': {
                'max_components_to_analyze': 100,
                'batch_size': 16
            },
            'reply_analysis': {
                'max_reply_depth': 5,
                'classification_batch_size': 16
            },
            'output': {
                'save_intermediate_results': True,
                'create_visualizations': True
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        return self.config.get(section, {})
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'data.data_dir')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, output_path: str):
        """Save current configuration to file."""
        output_file = Path(output_path)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif output_file.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported output format: {output_file.suffix}")
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
    
    def validate_config(self) -> bool:
        """Validate the configuration structure and values."""
        required_sections = ['data', 'similarity']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate data section
        data_config = self.config['data']
        if 'data_dir' not in data_config:
            logger.error("Missing 'data_dir' in data configuration")
            return False
        
        if 'platforms' not in data_config:
            logger.error("Missing 'platforms' in data configuration")
            return False
        
        # Validate similarity section
        similarity_config = self.config['similarity']
        if 'threshold' in similarity_config:
            threshold = similarity_config['threshold']
            if not (0 <= threshold <= 1):
                logger.error(f"Similarity threshold must be between 0 and 1, got: {threshold}")
                return False
        
        logger.info("Configuration validation passed")
        return True
