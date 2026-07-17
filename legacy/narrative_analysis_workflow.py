#!/usr/bin/env python3
"""
Complete workflow for cross-platform narrative divergence analysis.
Integrates similarity graph construction with narrative frame analysis.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from similarity_graph import SimilarityGraphBuilder
from narrative_divergence_analyzer import NarrativeDivergenceAnalyzer

class NarrativeAnalysisWorkflow:
    """
    Complete workflow for analyzing narrative divergence across social media platforms.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 model_name: str = "all-MiniLM-L6-v2",
                 output_dir: str = "results"):
        """
        Initialize the workflow.
        
        Args:
            similarity_threshold: Threshold for semantic similarity matching
            model_name: Sentence transformer model name
            output_dir: Directory to save results
        """
        self.similarity_builder = SimilarityGraphBuilder(
            model_name=model_name, 
            similarity_threshold=similarity_threshold
        )
        self.narrative_analyzer = NarrativeDivergenceAnalyzer(model_name=model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_platform_data(self, data_path: str) -> Dict[str, List[str]]:
        """
        Load platform data from file.
        
        Args:
            data_path: Path to data file (JSON, CSV, or pickle)
            
        Returns:
            Dictionary mapping platform names to post lists
        """
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            # Assume columns: 'platform', 'post_text'
            platform_data = {}
            for platform in df['platform'].unique():
                platform_posts = df[df['platform'] == platform]['post_text'].tolist()
                platform_data[platform] = platform_posts
            return platform_data
        elif data_path.suffix == '.pkl':
            return pd.read_pickle(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def run_complete_analysis(self, 
                            platform_data: Dict[str, List[str]], 
                            experiment_name: str = "narrative_analysis") -> Dict:
        """
        Run the complete narrative divergence analysis workflow.
        
        Args:
            platform_data: Dictionary mapping platform names to post lists
            experiment_name: Name for this analysis run
            
        Returns:
            Dictionary containing all results
        """
        print(f"Starting narrative divergence analysis: {experiment_name}")
        print(f"Platform data: {[(platform, len(posts)) for platform, posts in platform_data.items()]}")
        
        # Step 1: Build similarity graph
        print("\n1. Building semantic similarity graph...")
        graph, offsets = self.similarity_builder.build_tripartite_graph(platform_data)
        
        # Step 2: Extract connected components
        print("2. Extracting connected components...")
        components = self.similarity_builder.extract_connected_components(
            graph, platform_data, offsets
        )
        print(f"Found {len(components)} semantic components")
        
        # Step 3: Analyze narrative divergence
        print("3. Analyzing narrative divergence...")
        ndi_results = self.narrative_analyzer.analyze_component_collection(components)
        print(f"Analyzed {len(ndi_results)} components with multiple platforms")
        
        # Step 4: Generate visualizations
        print("4. Generating visualizations...")
        viz_path = self.output_dir / f"{experiment_name}_visualizations.png"
        self.narrative_analyzer.visualize_ndi_results(ndi_results, str(viz_path))
        
        # Step 5: Generate summary report
        print("5. Generating summary report...")
        summary_report = self.narrative_analyzer.generate_summary_report(ndi_results)
        
        # Save all results
        results = {
            'experiment_name': experiment_name,
            'platform_data_summary': {
                platform: {
                    'post_count': len(posts),
                    'avg_post_length': np.mean([len(post.split()) for post in posts])
                }
                for platform, posts in platform_data.items()
            },
            'graph_stats': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'connected_components': len(components)
            },
            'ndi_results': ndi_results.to_dict('records'),
            'summary_report': summary_report
        }
        
        # Save results to files
        self._save_results(results, experiment_name)
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
        print("\nSummary Report:")
        print(summary_report)
        
        return results
    
    def _save_results(self, results: Dict, experiment_name: str):
        """Save results to various file formats."""
        
        # Save main results as JSON
        results_path = self.output_dir / f"{experiment_name}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save NDI results as CSV
        ndi_df = pd.DataFrame(results['ndi_results'])
        csv_path = self.output_dir / f"{experiment_name}_ndi_metrics.csv"
        ndi_df.to_csv(csv_path, index=False)
        
        # Save summary report as text
        report_path = self.output_dir / f"{experiment_name}_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(results['summary_report'])
    
    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """
        Compare results across multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            
        Returns:
            DataFrame comparing key metrics across experiments
        """
        comparison_data = []
        
        for exp_name in experiment_names:
            results_path = self.output_dir / f"{exp_name}_results.json"
            if not results_path.exists():
                print(f"Warning: Results file not found for {exp_name}")
                continue
                
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            ndi_df = pd.DataFrame(results['ndi_results'])
            
            # Calculate summary statistics
            js_cols = [col for col in ndi_df.columns if 'js_divergence' in col]
            centroid_cols = [col for col in ndi_df.columns if 'centroid_distance' in col]
            
            summary = {
                'experiment': exp_name,
                'components_analyzed': len(ndi_df),
                'total_posts': ndi_df['total_posts'].sum(),
                'avg_posts_per_component': ndi_df['total_posts'].mean(),
                'avg_js_divergence': ndi_df[js_cols].mean().mean() if js_cols else np.nan,
                'avg_centroid_distance': ndi_df[centroid_cols].mean().mean() if centroid_cols else np.nan,
                'nodes': results['graph_stats']['nodes'],
                'edges': results['graph_stats']['edges']
            }
            
            comparison_data.append(summary)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comp_path = self.output_dir / "experiment_comparison.csv"
        comparison_df.to_csv(comp_path, index=False)
        
        return comparison_df

def create_sample_platform_data() -> Dict[str, List[str]]:
    """Create sample data for testing."""
    return {
        'truth': [
            "This indictment is a complete witch hunt by the deep state. They're trying to stop Trump because they know he'll win.",
            "Another bogus charge against the greatest president we've ever had. The people see right through this.",
            "Trump is being persecuted for fighting for America. This is what happens when you threaten the establishment.",
            "They're weaponizing the justice system against their political opponents. Banana republic tactics.",
            "The deep state and fake media are working together to destroy Trump because he exposes their corruption."
        ],
        'bluesky': [
            "Oh great, another Trump indictment. At this point it's just exhausting political theater.",
            "This whole circus is making a mockery of our institutions. Both sides are playing games.",
            "The system is so broken. Whether you support Trump or not, this constant drama helps nobody.",
            "I'm so tired of this endless cycle. Can we please focus on actual policy issues?",
            "The whole political establishment needs to be reformed. This back-and-forth accomplishes nothing."
        ],
        'mastodon': [
            "Finally, some accountability. No one should be above the law, regardless of their position.",
            "This is what justice looks like in a functioning democracy. The rule of law must be upheld.",
            "It's crucial that we hold all leaders accountable for their actions. Democracy depends on it.",
            "Justice delayed is justice denied. These charges should have been brought years ago.",
            "This shows our democratic institutions are working, even when it's politically difficult."
        ]
    }

def main():
    """Main function for running the workflow."""
    # Initialize workflow
    workflow = NarrativeAnalysisWorkflow(
        similarity_threshold=0.7,
        model_name="all-MiniLM-L6-v2",
        output_dir="results"
    )
    
    # Option 1: Use sample data
    print("Running analysis with sample data...")
    sample_data = create_sample_platform_data()
    results = workflow.run_complete_analysis(sample_data, "sample_analysis")
    
    # Option 2: Load data from file (uncomment to use)
    # print("Loading data from file...")
    # platform_data = workflow.load_platform_data("data/platform_posts.json")
    # results = workflow.run_complete_analysis(platform_data, "real_data_analysis")
    
    print("\nWorkflow completed successfully!")
    return results

if __name__ == "__main__":
    results = main()
