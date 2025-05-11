"""Reply chain analysis module for platform narratives analysis.

This module provides tools for analyzing reply chains in social media conversations
to determine how replies relate to the original post's narrative. It works with the
NarrativeTreeAnalyzer to provide deeper insights into conversation dynamics.
"""

from typing import List, Dict, Any, Optional, Union

class ReplyChainAnalyzer:
    """
    Analyzes reply chains to determine how replies relate to the original post's narrative.
    
    This class provides methods for analyzing whether replies reinforce, challenge, or shift
    the narrative of a root post. It can be used with any LLM pipeline that accepts messages
    in the format expected by OpenAI's API.
    """
    
    def __init__(self, llm_pipeline):
        """
        Initialize the reply chain analyzer.
        
        Args:
            llm_pipeline: A callable that takes messages and returns generated text
                         Should accept the format used by OpenAI's API
        """
        self.pipe = llm_pipeline
    
    def analyze_reply_chain(self, root_post: str, replies: List[str]) -> Dict[str, Any]:
        """
        Analyze how replies relate to the root post's narrative.
        
        Args:
            root_post: Text content of the original post
            replies: List of reply text content
            
        Returns:
            Dictionary with analysis results including classifications
        """
        prompt = f"""
        You are analyzing an online conversation.

        Root Post:
        {root_post}

        Replies:
        {replies}

        Instructions:
        Identify whether replies reinforce, challenge, or shift the narrative of the root post.

        - Reinforce: The reply supports Trump and echoes the claim that The Washington Post is "Fake News."

        - Challenge: The reply supports the reporting by The Washington Post and disagrees with labeling it as "Fake News."

        - Shift: The reply introduces a new perspective or topic that is not directly related to Trump or The Washington Post, such as media in general, unrelated political issues, or broader societal commentary.

        ONLY output one category from: ["reinforce", "challenge", "shift"].
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        output = self.pipe(messages, max_new_tokens=200)
        response_text = output[0]["generated_text"][-1]["content"].strip().lower()
        
        # Validate response is one of the expected categories
        valid_categories = ["reinforce", "challenge", "shift"]
        category = next((cat for cat in valid_categories if cat in response_text), "unknown")
        
        return {
            "category": category,
            "raw_response": response_text
        }
    
    def batch_analyze_reply_chains(self, data, batch_size: int = 8):
        """
        Analyze multiple reply chains in batches.
        
        Args:
            data: List of dictionaries, each containing 'root_post' and 'replies' keys
            batch_size: Number of chains to process in each batch
            
        Returns:
            List of analysis results by chain
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_results = []
            
            for item in batch:
                root_post = item.get('root_post', '')
                replies = item.get('replies', [])
                result = self.analyze_reply_chain(root_post, replies)
                batch_results.append(result)
            
            all_results.extend(batch_results)
            
        return all_results
    
    def get_narrative_dynamics_summary(self, results):
        """
        Generate a summary of narrative dynamics from analysis results.
        
        Args:
            results: List of analysis results from analyze_reply_chain
            
        Returns:
            Dictionary with summary statistics
        """
        # Count occurrences of each category
        categories = [result.get('category', 'unknown') for result in results]
        category_counts = {
            'reinforce': categories.count('reinforce'),
            'challenge': categories.count('challenge'),
            'shift': categories.count('shift'),
            'unknown': categories.count('unknown')
        }
        
        # Calculate percentages
        total = len(results)
        category_percentages = {}
        if total > 0:
            for category, count in category_counts.items():
                category_percentages[category] = round((count / total) * 100, 2)
        
        return {
            'counts': category_counts,
            'percentages': category_percentages,
            'total_analyzed': total
        }