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
        Classify a single post's narrative frame probabilities.
        
        Args:
            post: Text content of the post to analyze
            
        Returns:
            Dictionary with frame probabilities (0-1 for each frame)
        """
        prompt = f"""You are a framing analyst working on political social media content.

Your task is to evaluate a given post and assign **narrative frame probabilities** across a predefined set of 10 frame categories. A single post may activate multiple frames to varying degrees.

For each frame, return a value between 0 and 1 (to two decimal places) representing how strongly the frame is present in the post. These values are independent and do not need to sum to 1.

### Frame categories and definitions:

1. **Persecution** — The speaker or group is unfairly targeted, censored, or attacked by elites or institutions.
2. **Corruption** — Highlights unethical, illegal, or immoral behavior by politicians or institutions.
3. **Accountability** — Emphasizes legal consequences, justice, or democratic responsibility.
4. **Irony/Detachment** — Uses sarcasm, memes, or emotional distance to comment on politics.
5. **Heroism** — Portrays someone as brave, self-sacrificing, or fighting for the people.
6. **Civic Critique** — Critiques systemic flaws in governance, democracy, or policy.
7. **Moral Decay** — Suggests societal decline in values, ethics, or tradition.
8. **Media Manipulation** — Claims mainstream media is biased, deceptive, or agenda-driven.
9. **Strategic Pragmatism** — Expresses resignation, lesser-evil voting, or political calculation.
10. **Cultural Identity** — Frames politics through national, ethnic, or group-based belonging.

### Output format (JSON-style dictionary):
Return a dictionary where keys are frame names and values are probabilities.

### Example:

Post:  
> "Of course the DOJ is going after him again right before the election. Meanwhile, Hunter walks free. Disgusting."

Your output:
```json
{{
  "Persecution": 0.85,
  "Corruption": 0.70,
  "Accountability": 0.10,
  "Irony/Detachment": 0.00,
  "Heroism": 0.00,
  "Civic Critique": 0.15,
  "Moral Decay": 0.40,
  "Media Manipulation": 0.60,
  "Strategic Pragmatism": 0.05,
  "Cultural Identity": 0.00
}}
```

Now classify the following post:

{post}"""

        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in political discourse framing analysis."
                },
                {
                    "role": "user",
                    "content":  prompt
                },
            ]

        output = self.pipe(messages, max_new_tokens=300)
        response_text = output[0]["generated_text"][-1]["content"]
        
        # Try to parse JSON response
        try:
            # Try to extract JSON from the response if it's wrapped in markdown
            json_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith("```json"):
                json_text = json_text[7:]  # Remove ```json
            elif json_text.startswith("```"):
                json_text = json_text[3:]   # Remove ```
            
            if json_text.endswith("```"):
                json_text = json_text[:-3]  # Remove trailing ```
            
            json_text = json_text.strip()
            
            frame_probabilities = json.loads(json_text)
            # Validate and clean the response
            return self._validate_frame_probabilities(frame_probabilities)
        except json.JSONDecodeError as e:
            # If parsing fails, return detailed error info
            return {
                "raw_response": response_text, 
                "error": "JSON parsing failed",
                "json_error": str(e),
                "cleaned_text": json_text if 'json_text' in locals() else response_text
            }
    
    def _validate_frame_probabilities(self, frame_probabilities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean frame probability results.
        
        Args:
            frame_probabilities: Raw frame probabilities from LLM
            
        Returns:
            Validated and cleaned frame probabilities
        """
        expected_frames = [
            "Persecution", "Corruption", "Accountability", "Irony/Detachment",
            "Heroism", "Civic Critique", "Moral Decay", "Media Manipulation",
            "Strategic Pragmatism", "Cultural Identity"
        ]
        
        validated = {}
        
        for frame in expected_frames:
            if frame in frame_probabilities:
                prob = frame_probabilities[frame]
                # Ensure probability is between 0 and 1
                if isinstance(prob, (int, float)):
                    validated[frame] = max(0.0, min(1.0, float(prob)))
                else:
                    validated[frame] = 0.0
            else:
                validated[frame] = 0.0
        
        # Add metadata
        validated["_metadata"] = {
            "total_frames": len(expected_frames),
            "detected_frames": len([f for f in expected_frames if f in frame_probabilities]),
            "avg_probability": sum(validated[f] for f in expected_frames) / len(expected_frames),
            "max_frame": max(expected_frames, key=lambda f: validated[f]),
            "max_probability": max(validated[f] for f in expected_frames)
        }
        
        return validated
    
    def classify_component_posts(self, component_posts: Dict[str, List[str]], 
                               max_posts_per_platform: int = 5) -> List[Dict[str, Any]]:
        """
        Classify posts from different platforms within a component using frame probabilities.
        
        Args:
            component_posts: Dictionary mapping platform names to lists of posts
            max_posts_per_platform: Maximum number of posts to analyze per platform
            
        Returns:
            List of dictionaries with frame probability results by platform
        """
        results = []
        
        for platform, posts in component_posts.items():
            # Skip if too many posts
            if len(posts) > max_posts_per_platform:
                continue
            
            platform_frames = []
            
            # Classify each post individually for better granularity
            for i, post in enumerate(posts[:max_posts_per_platform]):
                if len(post.strip()) > 10:  # Skip very short posts
                    frame_probs = self.classify_post(post)
                    frame_probs["post_index"] = i
                    frame_probs["post_preview"] = post[:100] + "..." if len(post) > 100 else post
                    platform_frames.append(frame_probs)
            
            if platform_frames:
                # Calculate aggregate statistics for the platform
                aggregate_stats = self._aggregate_platform_frames(platform_frames)
                
                results.append({
                    "platform": platform,
                    "num_posts": len(platform_frames),
                    "individual_classifications": platform_frames,
                    "aggregate_frames": aggregate_stats
                })
            
        return results
    
    def _aggregate_platform_frames(self, platform_frames: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate frame probabilities across multiple posts from the same platform.
        
        Args:
            platform_frames: List of frame probability dictionaries
            
        Returns:
            Aggregated frame probabilities and statistics
        """
        expected_frames = [
            "Persecution", "Corruption", "Accountability", "Irony/Detachment",
            "Heroism", "Civic Critique", "Moral Decay", "Media Manipulation",
            "Strategic Pragmatism", "Cultural Identity"
        ]
        
        # Filter out posts with errors
        valid_frames = [f for f in platform_frames if "error" not in f and "_metadata" in f]
        
        if not valid_frames:
            return {frame: 0.0 for frame in expected_frames}
        
        # Calculate mean probabilities across posts
        aggregated = {}
        for frame in expected_frames:
            probabilities = [f.get(frame, 0.0) for f in valid_frames]
            aggregated[f"{frame}_mean"] = sum(probabilities) / len(probabilities)
            aggregated[f"{frame}_max"] = max(probabilities)
            aggregated[f"{frame}_min"] = min(probabilities)
        
        # Add summary statistics
        aggregated["_summary"] = {
            "valid_posts": len(valid_frames),
            "total_posts": len(platform_frames),
            "dominant_frame": max(expected_frames, key=lambda f: aggregated[f"{f}_mean"]),
            "frame_diversity": len([f for f in expected_frames if aggregated[f"{f}_mean"] > 0.1])
        }
        
        return aggregated
    
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
    
    def calculate_narrative_divergence(self, platform_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate narrative divergence between platforms using frame probabilities.
        
        Args:
            platform_results: Results from classify_component_posts
            
        Returns:
            Dictionary with divergence metrics and analysis
        """
        import numpy as np
        
        if len(platform_results) < 2:
            return {"error": "Need at least 2 platforms for divergence calculation"}
        
        expected_frames = [
            "Persecution", "Corruption", "Accountability", "Irony/Detachment",
            "Heroism", "Civic Critique", "Moral Decay", "Media Manipulation",
            "Strategic Pragmatism", "Cultural Identity"
        ]
        
        # Extract mean frame probabilities for each platform
        platform_vectors = {}
        for result in platform_results:
            platform = result["platform"]
            if "aggregate_frames" in result:
                vector = [result["aggregate_frames"].get(f"{frame}_mean", 0.0) for frame in expected_frames]
                platform_vectors[platform] = np.array(vector)
        
        if len(platform_vectors) < 2:
            return {"error": "Insufficient valid platform data for divergence calculation"}
        
        # Calculate pairwise divergences
        platforms = list(platform_vectors.keys())
        divergences = {}
        
        for i in range(len(platforms)):
            for j in range(i + 1, len(platforms)):
                platform1, platform2 = platforms[i], platforms[j]
                vec1, vec2 = platform_vectors[platform1], platform_vectors[platform2]
                
                # Hellinger distance (good for probability distributions)
                hellinger = np.sqrt(np.sum((np.sqrt(vec1) - np.sqrt(vec2)) ** 2)) / np.sqrt(2)
                
                # Cosine distance
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                cosine_distance = 1 - cosine_sim
                
                # Jensen-Shannon divergence approximation
                m = (vec1 + vec2) / 2
                js_div = 0.5 * np.sum(vec1 * np.log(vec1 / (m + 1e-8) + 1e-8)) + \
                        0.5 * np.sum(vec2 * np.log(vec2 / (m + 1e-8) + 1e-8))
                js_div = max(0, js_div)  # Ensure non-negative
                
                divergences[f"{platform1}_vs_{platform2}"] = {
                    "hellinger_distance": float(hellinger),
                    "cosine_distance": float(cosine_distance),
                    "js_divergence": float(js_div),
                    "frame_differences": {
                        frame: float(abs(vec1[i] - vec2[i])) 
                        for i, frame in enumerate(expected_frames)
                    }
                }
        
        # Overall analysis
        all_hellinger = [d["hellinger_distance"] for d in divergences.values()]
        
        return {
            "pairwise_divergences": divergences,
            "summary": {
                "avg_hellinger_distance": float(np.mean(all_hellinger)),
                "max_hellinger_distance": float(np.max(all_hellinger)),
                "min_hellinger_distance": float(np.min(all_hellinger)),
                "most_divergent_pair": max(divergences.keys(), key=lambda k: divergences[k]["hellinger_distance"]),
                "platforms_analyzed": platforms,
                "num_comparisons": len(divergences)
            }
        }
    
    def get_frame_distribution_summary(self, platform_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of frame distributions across platforms.
        
        Args:
            platform_results: Results from classify_component_posts
            
        Returns:
            Summary of frame patterns and dominant narratives
        """
        expected_frames = [
            "Persecution", "Corruption", "Accountability", "Irony/Detachment",
            "Heroism", "Civic Critique", "Moral Decay", "Media Manipulation",
            "Strategic Pragmatism", "Cultural Identity"
        ]
        
        summary = {
            "platform_summaries": {},
            "cross_platform_analysis": {}
        }
        
        # Analyze each platform
        for result in platform_results:
            platform = result["platform"]
            if "aggregate_frames" in result:
                agg = result["aggregate_frames"]
                
                # Find dominant frames (mean > 0.3)
                dominant_frames = [
                    frame for frame in expected_frames 
                    if agg.get(f"{frame}_mean", 0) > 0.3
                ]
                
                # Find secondary frames (mean > 0.1)
                secondary_frames = [
                    frame for frame in expected_frames 
                    if 0.1 < agg.get(f"{frame}_mean", 0) <= 0.3
                ]
                
                summary["platform_summaries"][platform] = {
                    "dominant_frames": dominant_frames,
                    "secondary_frames": secondary_frames,
                    "frame_diversity": agg.get("_summary", {}).get("frame_diversity", 0),
                    "top_frame": agg.get("_summary", {}).get("dominant_frame", "unknown"),
                    "num_posts": result.get("num_posts", 0)
                }
        
        # Cross-platform analysis
        all_dominant = []
        all_secondary = []
        for platform_data in summary["platform_summaries"].values():
            all_dominant.extend(platform_data["dominant_frames"])
            all_secondary.extend(platform_data["secondary_frames"])
        
        # Count frame frequency across platforms
        from collections import Counter
        dominant_counts = Counter(all_dominant)
        secondary_counts = Counter(all_secondary)
        
        summary["cross_platform_analysis"] = {
            "most_common_dominant_frames": dominant_counts.most_common(5),
            "most_common_secondary_frames": secondary_counts.most_common(5),
            "unique_frames_used": len(set(all_dominant + all_secondary)),
            "total_platforms": len(summary["platform_summaries"])
        }
        
        return summary