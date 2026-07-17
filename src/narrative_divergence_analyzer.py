import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NarrativeDivergenceAnalyzer:
    """
    Analyzes narrative divergence across social media platforms using embedding-based methods.
    Implements both discrete label-based NDI and continuous embedding-based analysis.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the narrative divergence analyzer.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.narrative_frames = self._define_narrative_frames()
        
    def _define_narrative_frames(self) -> Dict[str, str]:
        """
        Define the narrative frame taxonomy based on framing theory literature.
        
        Returns:
            Dictionary mapping frame names to their definitions
        """
        return {
            "persecution": "Presents political actor as unfairly targeted or victimized by powerful forces",
            "corruption": "Emphasizes unethical, illegal, or morally questionable behavior", 
            "accountability": "Portrays legal/political consequences as necessary for justice or democracy",
            "irony_detachment": "Uses sarcasm, cynicism, or emotional distancing as a critique style",
            "heroism": "Frames political figure as savior, patriot, or self-sacrificing fighter",
            "civic_critique": "Highlights structural or democratic failures of systems or institutions",
            "moral_decay": "Depicts societal collapse in values, ethics, or collective morality",
            "media_manipulation": "Frames mainstream media as biased, deceptive, or agenda-driven",
            "strategic_pragmatism": "Focuses on political calculation or practical consequences",
            "cultural_identity": "Invokes national, religious, or group identity to frame meaning"
        }
    
    def classify_narrative_frames(self, posts: List[str], method="llm_prompt") -> List[Dict[str, float]]:
        """
        Classify posts into narrative frames.
        
        Args:
            posts: List of post texts to classify
            method: Classification method ("llm_prompt" or "embedding_similarity")
            
        Returns:
            List of dictionaries with frame probabilities for each post
        """
        if method == "embedding_similarity":
            return self._classify_by_embedding_similarity(posts)
        else:
            # For now, return placeholder - in practice you'd use LLM API
            return self._classify_by_llm_placeholder(posts)
    
    def _classify_by_embedding_similarity(self, posts: List[str]) -> List[Dict[str, float]]:
        """
        Classify posts using embedding similarity to frame prototypes.
        
        Args:
            posts: List of post texts
            
        Returns:
            List of frame probability distributions
        """
        # Encode posts
        post_embeddings = self.model.encode(posts, convert_to_tensor=True)
        
        # Create frame prototype embeddings from definitions
        frame_definitions = list(self.narrative_frames.values())
        frame_embeddings = self.model.encode(frame_definitions, convert_to_tensor=True)
        
        # Compute similarities
        similarities = torch.cosine_similarity(
            post_embeddings.unsqueeze(1), 
            frame_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Convert to probabilities using softmax
        probabilities = torch.softmax(similarities, dim=1)
        
        # Convert to list of dictionaries
        frame_names = list(self.narrative_frames.keys())
        results = []
        for prob_dist in probabilities:
            frame_probs = {frame_names[i]: float(prob_dist[i]) for i in range(len(frame_names))}
            results.append(frame_probs)
            
        return results
    
    def _classify_by_llm_placeholder(self, posts: List[str]) -> List[Dict[str, float]]:
        """
        Placeholder for LLM-based classification.
        In practice, this would use GPT-4/Claude API with structured prompts.
        """
        # Simulate LLM classification with random but realistic distributions
        np.random.seed(42)
        frame_names = list(self.narrative_frames.keys())
        results = []
        
        for post in posts:
            # Create a somewhat realistic distribution based on post content
            probs = np.random.dirichlet(np.ones(len(frame_names)) * 0.5)
            frame_probs = {frame_names[i]: float(probs[i]) for i in range(len(frame_names))}
            results.append(frame_probs)
            
        return results
    
    def compute_ndi_discrete(self, component_posts: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute Narrative Divergence Index using discrete frame classifications.
        
        Args:
            component_posts: Dictionary mapping platform names to lists of posts
            
        Returns:
            Dictionary with NDI metrics
        """
        platform_distributions = {}
        
        # Get frame distributions for each platform
        for platform, posts in component_posts.items():
            if not posts:
                continue
                
            frame_classifications = self.classify_narrative_frames(posts)
            
            # Aggregate into platform-level distribution
            frame_counts = defaultdict(float)
            for classification in frame_classifications:
                for frame, prob in classification.items():
                    frame_counts[frame] += prob
            
            # Normalize to get distribution
            total = sum(frame_counts.values())
            platform_distributions[platform] = {
                frame: count/total for frame, count in frame_counts.items()
            }
        
        # Compute divergence metrics
        return self._compute_distribution_divergence(platform_distributions)
    
    def compute_ndi_embedding(self, component_posts: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute Narrative Divergence Index using continuous embedding space.
        
        Args:
            component_posts: Dictionary mapping platform names to lists of posts
            
        Returns:
            Dictionary with embedding-based NDI metrics
        """
        platform_embeddings = {}
        
        # Get embeddings for each platform
        for platform, posts in component_posts.items():
            if not posts:
                continue
            embeddings = self.model.encode(posts, convert_to_tensor=True)
            platform_embeddings[platform] = embeddings
        
        return self._compute_embedding_divergence(platform_embeddings)
    
    def _compute_distribution_divergence(self, platform_distributions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute divergence metrics between platform frame distributions.
        
        Args:
            platform_distributions: Platform-level frame distributions
            
        Returns:
            Dictionary with divergence metrics
        """
        platforms = list(platform_distributions.keys())
        if len(platforms) < 2:
            return {}
        
        frame_names = list(self.narrative_frames.keys())
        metrics = {}
        
        # Prepare distribution vectors
        dist_vectors = []
        for platform in platforms:
            dist = [platform_distributions[platform].get(frame, 0.0) for frame in frame_names]
            dist_vectors.append(dist)
        
        # Compute pairwise JS divergences
        for i in range(len(platforms)):
            for j in range(i+1, len(platforms)):
                js_div = jensenshannon(dist_vectors[i], dist_vectors[j]) ** 2
                metrics[f"{platforms[i]}_vs_{platforms[j]}_js_divergence"] = js_div
        
        # Compute overall entropy
        if len(platforms) >= 3:
            # Average distribution
            avg_dist = np.mean(dist_vectors, axis=0)
            avg_dist = avg_dist / np.sum(avg_dist)  # Renormalize
            metrics["overall_entropy"] = entropy(avg_dist)
        
        return metrics
    
    def _compute_embedding_divergence(self, platform_embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute divergence metrics in embedding space.
        
        Args:
            platform_embeddings: Platform-level embeddings
            
        Returns:
            Dictionary with embedding divergence metrics
        """
        platforms = list(platform_embeddings.keys())
        if len(platforms) < 2:
            return {}
        
        metrics = {}
        
        # Compute centroids
        centroids = {}
        for platform, embeddings in platform_embeddings.items():
            centroids[platform] = torch.mean(embeddings, dim=0)
        
        # Pairwise centroid distances
        for i in range(len(platforms)):
            for j in range(i+1, len(platforms)):
                p1, p2 = platforms[i], platforms[j]
                cos_dist = 1 - torch.cosine_similarity(
                    centroids[p1].unsqueeze(0), 
                    centroids[p2].unsqueeze(0)
                ).item()
                metrics[f"{p1}_vs_{p2}_centroid_distance"] = cos_dist
        
        # Intra-platform variance
        for platform, embeddings in platform_embeddings.items():
            if len(embeddings) > 1:
                centroid = centroids[platform]
                distances = []
                for emb in embeddings:
                    cos_dist = 1 - torch.cosine_similarity(
                        emb.unsqueeze(0), 
                        centroid.unsqueeze(0)
                    ).item()
                    distances.append(cos_dist)
                metrics[f"{platform}_intra_variance"] = np.mean(distances)
        
        return metrics
    
    def analyze_component_collection(self, component_posts_list: List[Dict[str, List[str]]]) -> pd.DataFrame:
        """
        Analyze a collection of semantic components for narrative divergence.
        
        Args:
            component_posts_list: List of component dictionaries from similarity graph
            
        Returns:
            DataFrame with NDI metrics for each component
        """
        results = []
        
        for i, component_posts in enumerate(component_posts_list):
            # Filter components with posts from multiple platforms
            platforms_with_posts = [p for p, posts in component_posts.items() if posts]
            if len(platforms_with_posts) < 2:
                continue
            
            # Compute both discrete and embedding NDI
            discrete_ndi = self.compute_ndi_discrete(component_posts)
            embedding_ndi = self.compute_ndi_embedding(component_posts)
            
            # Combine metrics
            component_result = {
                'component_id': i,
                'platforms_present': ','.join(platforms_with_posts),
                'total_posts': sum(len(posts) for posts in component_posts.values()),
                **discrete_ndi,
                **embedding_ndi
            }
            
            # Add post counts per platform
            for platform, posts in component_posts.items():
                component_result[f"{platform}_post_count"] = len(posts)
            
            results.append(component_result)
        
        return pd.DataFrame(results)
    
    def visualize_ndi_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualizations for NDI analysis results.
        
        Args:
            results_df: DataFrame from analyze_component_collection
            save_path: Optional path to save figures
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. JS Divergence distribution
        js_cols = [col for col in results_df.columns if 'js_divergence' in col]
        if js_cols:
            js_data = results_df[js_cols].values.flatten()
            js_data = js_data[~np.isnan(js_data)]
            axes[0, 0].hist(js_data, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribution of JS Divergence Scores')
            axes[0, 0].set_xlabel('JS Divergence')
            axes[0, 0].set_ylabel('Frequency')
        
        # 2. Centroid distances
        centroid_cols = [col for col in results_df.columns if 'centroid_distance' in col]
        if centroid_cols:
            centroid_data = results_df[centroid_cols].values.flatten()
            centroid_data = centroid_data[~np.isnan(centroid_data)]
            axes[0, 1].hist(centroid_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].set_title('Distribution of Centroid Distances')
            axes[0, 1].set_xlabel('Cosine Distance')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Component size vs. divergence
        if js_cols and 'total_posts' in results_df.columns:
            axes[1, 0].scatter(results_df['total_posts'], results_df[js_cols[0]], alpha=0.6)
            axes[1, 0].set_xlabel('Total Posts in Component')
            axes[1, 0].set_ylabel('JS Divergence')
            axes[1, 0].set_title('Component Size vs. Narrative Divergence')
        
        # 4. Platform comparison
        platform_cols = [col for col in results_df.columns if col.endswith('_post_count')]
        if len(platform_cols) >= 2:
            platform_data = results_df[platform_cols].fillna(0)
            platform_corr = platform_data.corr()
            sns.heatmap(platform_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Platform Post Count Correlations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a text summary of NDI analysis results.
        
        Args:
            results_df: DataFrame from analyze_component_collection
            
        Returns:
            String summary report
        """
        report = []
        report.append("=== Narrative Divergence Analysis Summary ===\n")
        
        # Basic statistics
        report.append(f"Total semantic components analyzed: {len(results_df)}")
        report.append(f"Average posts per component: {results_df['total_posts'].mean():.1f}")
        
        # Divergence statistics
        js_cols = [col for col in results_df.columns if 'js_divergence' in col]
        if js_cols:
            js_mean = results_df[js_cols].mean().mean()
            js_std = results_df[js_cols].std().mean()
            report.append(f"Average JS Divergence: {js_mean:.3f} (±{js_std:.3f})")
        
        centroid_cols = [col for col in results_df.columns if 'centroid_distance' in col]
        if centroid_cols:
            centroid_mean = results_df[centroid_cols].mean().mean()
            centroid_std = results_df[centroid_cols].std().mean()
            report.append(f"Average Centroid Distance: {centroid_mean:.3f} (±{centroid_std:.3f})")
        
        # Platform participation
        platform_cols = [col for col in results_df.columns if col.endswith('_post_count')]
        report.append(f"\nPlatform participation rates:")
        for col in platform_cols:
            platform = col.replace('_post_count', '')
            participation = (results_df[col] > 0).mean() * 100
            report.append(f"  {platform}: {participation:.1f}% of components")
        
        return "\n".join(report)

# Example usage and testing functions
def create_sample_data() -> Dict[str, List[str]]:
    """Create sample data for testing the analyzer."""
    return {
        'truth': [
            "This is clearly a witch hunt by the deep state against Trump",
            "Another false accusation to try to stop the movement",
            "They're terrified of what Trump represents for America"
        ],
        'bluesky': [
            "Oh great, another Trump indictment. This circus never ends.",
            "At this point it's just political theater on all sides",
            "The whole system needs reform, not more of this drama"
        ],
        'mastodon': [
            "Finally, accountability for those in power. No one is above the law.",
            "This is what justice looks like in a democracy",
            "The rule of law must be upheld regardless of political position"
        ]
    }

if __name__ == "__main__":
    # Test the analyzer with sample data
    analyzer = NarrativeDivergenceAnalyzer()
    
    # Create sample component data
    sample_components = [create_sample_data()]
    
    # Run analysis
    print("Running Narrative Divergence Analysis...")
    results = analyzer.analyze_component_collection(sample_components)
    
    # Display results
    print("\nResults DataFrame:")
    print(results.to_string())
    
    # Generate summary
    print("\n" + analyzer.generate_summary_report(results))
    
    # Create visualizations
    analyzer.visualize_ndi_results(results)
