"""Narrative classification module for platform narratives analysis.

This module provides tools for classifying narratives in social media posts
using LLM-based analysis. It serves as a component in the platform narratives
analysis pipeline, working alongside FastLexRank and SimilarityGraphBuilder.
"""

import json
from typing import List, Dict, Any, Optional, Union
from tqdm.auto import tqdm

class NarrativeClassifier:
    """
    Classifies narratives in social media posts using LLM-based analysis.
    
    This class provides methods for analyzing posts to identify narrative frames,
    subjects, stances, and topic focus. It can be used with any LLM pipeline that
    accepts messages in the format expected by OpenAI's API.
    """
    
    def __init__(self, llm_pipeline):
        """
        Initialize the narrative classifier.
        
        Args:
            llm_pipeline: A callable that takes messages and returns generated text
                         Should accept the format used by OpenAI's API
        """
        self.pipe = llm_pipeline
    
    def classify_post(self, post: str) -> Dict[str, Any]:
        """
        Classify a single post's narrative elements.
        
        Args:
            post: Text content of the post to analyze
            
        Returns:
            Dictionary with classification results
        """
        prompt = f"""
            You are a political discourse analyst.

            Classify the following social media post based on:

            1. Narrative Frame (e.g., corruption, persecution, legal justice, media bias, systemic inequality, etc.)
            2. Main Subject (e.g., Trump, Biden, etc.)
            3. Stance toward main subject (e.g., supportive, critical, neutral, unclear)
            4. Topic Focus (e.g., legal, cultural, institutional, personal attack)

            Return in JSON format.

            Post:
            {post}
            """

        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content":  prompt
                },
            ]

        output = self.pipe(messages, max_new_tokens=200)
        response_text = output[0]["generated_text"][-1]["content"]
        
        # Try to parse JSON response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, return the raw text
            return {"raw_response": response_text}
    
    def classify_component_posts(self, component_posts: Dict[str, List[str]], 
                               max_posts_per_platform: int = 10) -> List[Dict[str, Any]]:
        """
        Classify posts from different platforms within a component.
        
        Args:
            component_posts: Dictionary mapping platform names to lists of posts
            max_posts_per_platform: Maximum number of posts to analyze per platform
            
        Returns:
            List of dictionaries with classification results by platform
        """
        results = []
        
        for platform, posts in component_posts.items():
            # Skip if too many posts
            if len(posts) > max_posts_per_platform:
                continue
                
            # Join posts for context
            combined_post = ".".join(posts)
            
            # Classify the combined post
            classification = self.classify_post(combined_post)
            
            # Add to results
            results.append({
                "platform": platform,
                "classification": classification
            })
            
        return results
    
    def batch_classify_components(self, components_df, batch_size: int = 8):
        """
        Classify posts from multiple components in batches.
        
        Args:
            components_df: DataFrame with component posts
            batch_size: Number of components to process in each batch
            
        Returns:
            List of classification results by component
        """
        all_results = []
        
        # Convert DataFrame to records for processing
        components = components_df.to_dict(orient="records")
        
        # Process in batches with progress bar
        for i, component in tqdm(enumerate(components)):
            component_results = self.classify_component_posts(component)
            all_results.append(component_results)
            
        return all_results