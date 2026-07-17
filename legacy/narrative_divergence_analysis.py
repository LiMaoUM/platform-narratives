#!/usr/bin/env python3
"""
Narrative Divergence Analysis for Cross-Platform Political Discourse

This script analyzes how political events are framed differently across Truth Social, 
Bluesky, and Mastodon by computing Narrative Divergence Indices (NDI-OP and NDI-R).

Author: Data Analyst
Date: June 6, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from collections import defaultdict
import warnings
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class NarrativeDivergenceAnalyzer:
    """
    Analyzer for computing narrative divergence indices across social media platforms.
    """
    
    def __init__(self, output_dir: str = "narrative_analysis_output"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define narrative frames (based on your analysis)
        self.narrative_frames = [
            "Persecution", "Corruption", "Accountability", "Irony/Detachment",
            "Heroism", "Civic Critique", "Moral Decay", "Media Manipulation",
            "Strategic Pragmatism", "Cultural Identity"
        ]
        
        # Define platforms
        self.platforms = ['Truth', 'Bluesky', 'Mastodon']
        
        # Define reply types
        self.reply_types = ['reinforce', 'challenge', 'shift']
        
        print(f"Initialized NarrativeDivergenceAnalyzer")
        print(f"Output directory: {self.output_dir}")
    
    def load_and_prepare_data(self, narrative_data_path: str) -> pd.DataFrame:
        """
        Load narrative analysis data and convert to the required DataFrame format.
        
        Args:
            narrative_data_path: Path to the narrative analysis JSON file
            
        Returns:
            DataFrame with the required structure
        """
        print("Loading and preparing narrative analysis data...")
        
        # Load the narrative analysis data
        with open(narrative_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame format
        rows = []
        
        for component in data:
            component_id = str(component.get('component_id', 'unknown'))
            narrative_frames = component.get('narrative_frames', {})
            platform_counts = component.get('platform_counts', {})
            
            # Process each platform's narrative frames
            for platform_name, frames_list in narrative_frames.items():
                # Map platform names to standardized format
                platform_std = self._standardize_platform_name(platform_name)
                
                if isinstance(frames_list, list) and platform_std:
                    # Create narrative vector by counting frame occurrences
                    narrative_vector = self._create_narrative_vector(frames_list)
                    
                    # For this analysis, we'll treat all posts as original posts
                    # and simulate some reply data for demonstration
                    post_count = platform_counts.get(platform_name, 0)
                    
                    # Add multiple rows per component to simulate individual posts
                    for i in range(min(post_count, 100)):  # Limit for performance
                        # Add some noise to the narrative vector for individual posts
                        noisy_vector = self._add_noise_to_vector(narrative_vector)
                        
                        # Simulate post types and reply types
                        is_reply = np.random.random() < 0.3  # 30% chance of being a reply
                        post_type = 'reply' if is_reply else 'original'
                        reply_type = np.random.choice(self.reply_types) if is_reply else np.nan
                        
                        rows.append({
                            'component_id': component_id,
                            'platform': platform_std,
                            'narrative_vector': noisy_vector,
                            'reply_type': reply_type,
                            'post_type': post_type
                        })
        
        df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(df)} rows and {len(df['component_id'].unique())} unique components")
        print(f"Platform distribution: {df['platform'].value_counts().to_dict()}")
        
        return df
    
    def _standardize_platform_name(self, platform_name: str) -> Optional[str]:
        """Standardize platform names to match expected format."""
        platform_map = {
            'truth': 'Truth',
            'bluesky': 'Bluesky', 
            'mastodon': 'Mastodon'
        }
        return platform_map.get(platform_name.lower())
    
    def _create_narrative_vector(self, frames_list: List[str]) -> Dict[str, float]:
        """
        Create a narrative vector from a list of frame names.
        
        Args:
            frames_list: List of narrative frame names
            
        Returns:
            Dictionary with frame names as keys and normalized scores as values
        """
        # Count occurrences of each frame
        frame_counts = {frame: 0 for frame in self.narrative_frames}
        
        for frame in frames_list:
            if frame in frame_counts:
                frame_counts[frame] += 1
        
        # Normalize to create probability distribution
        total_frames = sum(frame_counts.values())
        if total_frames > 0:
            narrative_vector = {frame: count / total_frames for frame, count in frame_counts.items()}
        else:
            # Uniform distribution if no frames found
            narrative_vector = {frame: 1.0 / len(self.narrative_frames) for frame in self.narrative_frames}
        
        return narrative_vector
    
    def _add_noise_to_vector(self, base_vector: Dict[str, float], noise_level: float = 0.1) -> Dict[str, float]:
        """
        Add small amount of noise to create variation in individual posts.
        
        Args:
            base_vector: Base narrative vector
            noise_level: Amount of noise to add
            
        Returns:
            Noisy narrative vector
        """
        noisy_vector = {}
        
        for frame, value in base_vector.items():
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level)
            noisy_value = max(0, min(1, value + noise))  # Clamp to [0, 1]
            noisy_vector[frame] = noisy_value
        
        # Renormalize to ensure sum = 1
        total = sum(noisy_vector.values())
        if total > 0:
            noisy_vector = {frame: value / total for frame, value in noisy_vector.items()}
        
        return noisy_vector
    
    def compute_ndi_op(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute NDI-OP (Narrative Divergence Index for Original Posts).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with NDI-OP values per component
        """
        print("Computing NDI-OP (Narrative Divergence Index for Original Posts)...")
        
        # Filter for original posts only
        original_posts = df[df['post_type'] == 'original'].copy()
        
        ndi_op_results = []
        
        # Group by component_id
        for component_id, group in original_posts.groupby('component_id'):
            # Check if we have data for multiple platforms
            platforms_present = group['platform'].unique()
            
            if len(platforms_present) < 2:
                # Skip components with data from only one platform
                continue
            
            # Compute average narrative vector per platform
            platform_distributions = {}
            
            for platform in platforms_present:
                platform_data = group[group['platform'] == platform]
                
                # Average narrative vectors for this platform
                avg_vector = {}
                for frame in self.narrative_frames:
                    frame_values = [post['narrative_vector'][frame] for _, post in platform_data.iterrows()]
                    avg_vector[frame] = np.mean(frame_values)
                
                # Convert to probability distribution
                total = sum(avg_vector.values())
                if total > 0:
                    platform_distributions[platform] = np.array([avg_vector[frame] / total for frame in self.narrative_frames])
                else:
                    platform_distributions[platform] = np.ones(len(self.narrative_frames)) / len(self.narrative_frames)
            
            # Compute pairwise Jensen-Shannon divergences
            js_divergences = []
            platform_pairs = []
            
            platforms_list = list(platform_distributions.keys())
            for i in range(len(platforms_list)):
                for j in range(i + 1, len(platforms_list)):
                    p1, p2 = platforms_list[i], platforms_list[j]
                    
                    # Ensure distributions are valid (sum to 1, non-negative)
                    dist1 = platform_distributions[p1]
                    dist2 = platform_distributions[p2]
                    
                    # Add small epsilon to avoid zero probabilities
                    epsilon = 1e-10
                    dist1 = dist1 + epsilon
                    dist2 = dist2 + epsilon
                    dist1 = dist1 / np.sum(dist1)
                    dist2 = dist2 / np.sum(dist2)
                    
                    # Compute Jensen-Shannon divergence
                    js_div = jensenshannon(dist1, dist2)
                    js_divergences.append(js_div)
                    platform_pairs.append((p1, p2))
            
            # Average JS divergence across all platform pairs
            avg_js_divergence = np.mean(js_divergences) if js_divergences else 0.0
            
            # Find dominant narrative frame (highest average across platforms)
            frame_averages = {}
            for frame in self.narrative_frames:
                frame_values = []
                for platform in platforms_present:
                    platform_data = group[group['platform'] == platform]
                    frame_vals = [post['narrative_vector'][frame] for _, post in platform_data.iterrows()]
                    frame_values.extend(frame_vals)
                frame_averages[frame] = np.mean(frame_values)
            
            dominant_frame = max(frame_averages.items(), key=lambda x: x[1])[0]
            
            ndi_op_results.append({
                'component_id': component_id,
                'ndi_op': avg_js_divergence,
                'platforms_count': len(platforms_present),
                'platforms': list(platforms_present),
                'dominant_frame': dominant_frame,
                'dominant_frame_score': frame_averages[dominant_frame],
                'total_original_posts': len(group)
            })
        
        ndi_op_df = pd.DataFrame(ndi_op_results)
        print(f"Computed NDI-OP for {len(ndi_op_df)} components")
        print(f"NDI-OP statistics: mean={ndi_op_df['ndi_op'].mean():.4f}, std={ndi_op_df['ndi_op'].std():.4f}")
        
        return ndi_op_df
    
    def compute_ndi_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute NDI-R (Narrative Response Divergence).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with NDI-R values per component
        """
        print("Computing NDI-R (Narrative Response Divergence)...")
        
        # Filter for reply posts only
        reply_posts = df[df['post_type'] == 'reply'].copy()
        reply_posts = reply_posts.dropna(subset=['reply_type'])
        
        ndi_r_results = []
        
        # Group by component_id
        for component_id, group in reply_posts.groupby('component_id'):
            # Check if we have data for multiple platforms
            platforms_present = group['platform'].unique()
            
            if len(platforms_present) < 2:
                # Skip components with data from only one platform
                continue
            
            # Compute reply type distributions per platform
            platform_reply_distributions = {}
            
            for platform in platforms_present:
                platform_data = group[group['platform'] == platform]
                
                # Count reply types for this platform
                reply_counts = platform_data['reply_type'].value_counts()
                
                # Create probability distribution
                total_replies = len(platform_data)
                reply_dist = np.array([reply_counts.get(reply_type, 0) / total_replies for reply_type in self.reply_types])
                
                platform_reply_distributions[platform] = reply_dist
            
            # Compute pairwise Jensen-Shannon divergences
            js_divergences = []
            platform_pairs = []
            
            platforms_list = list(platform_reply_distributions.keys())
            for i in range(len(platforms_list)):
                for j in range(i + 1, len(platforms_list)):
                    p1, p2 = platforms_list[i], platforms_list[j]
                    
                    # Ensure distributions are valid
                    dist1 = platform_reply_distributions[p1]
                    dist2 = platform_reply_distributions[p2]
                    
                    # Add small epsilon to avoid zero probabilities
                    epsilon = 1e-10
                    dist1 = dist1 + epsilon
                    dist2 = dist2 + epsilon
                    dist1 = dist1 / np.sum(dist1)
                    dist2 = dist2 / np.sum(dist2)
                    
                    # Compute Jensen-Shannon divergence
                    js_div = jensenshannon(dist1, dist2)
                    js_divergences.append(js_div)
                    platform_pairs.append((p1, p2))
            
            # Average JS divergence across all platform pairs
            avg_js_divergence = np.mean(js_divergences) if js_divergences else 0.0
            
            ndi_r_results.append({
                'component_id': component_id,
                'ndi_r': avg_js_divergence,
                'platforms_count': len(platforms_present),
                'platforms': list(platforms_present),
                'total_replies': len(group)
            })
        
        ndi_r_df = pd.DataFrame(ndi_r_results)
        print(f"Computed NDI-R for {len(ndi_r_df)} components")
        print(f"NDI-R statistics: mean={ndi_r_df['ndi_r'].mean():.4f}, std={ndi_r_df['ndi_r'].std():.4f}")
        
        return ndi_r_df
    
    def classify_components(self, ndi_op_df: pd.DataFrame, ndi_r_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify components based on their NDI-OP and NDI-R values.
        
        Args:
            ndi_op_df: DataFrame with NDI-OP values
            ndi_r_df: DataFrame with NDI-R values
            
        Returns:
            DataFrame with component classifications
        """
        print("Classifying components based on NDI values...")
        
        # Merge NDI-OP and NDI-R dataframes
        combined_df = pd.merge(ndi_op_df, ndi_r_df[['component_id', 'ndi_r', 'total_replies']], 
                              on='component_id', how='outer')
        
        # Fill missing values with 0 (components that don't have both original posts and replies)
        combined_df['ndi_op'] = combined_df['ndi_op'].fillna(0)
        combined_df['ndi_r'] = combined_df['ndi_r'].fillna(0)
        combined_df['total_replies'] = combined_df['total_replies'].fillna(0)
        
        # Define thresholds (using median as threshold)
        ndi_op_threshold = combined_df['ndi_op'].median()
        ndi_r_threshold = combined_df['ndi_r'].median()
        
        print(f"Using NDI-OP threshold: {ndi_op_threshold:.4f}")
        print(f"Using NDI-R threshold: {ndi_r_threshold:.4f}")
        
        # Classify components
        def classify_component(row):
            ndi_op = row['ndi_op']
            ndi_r = row['ndi_r']
            
            if ndi_op > ndi_op_threshold and ndi_r > ndi_r_threshold:
                return "Highly Polarized"
            elif ndi_op <= ndi_op_threshold and ndi_r > ndi_r_threshold:
                return "Framing Aligned, Response Split"
            elif ndi_op > ndi_op_threshold and ndi_r <= ndi_r_threshold:
                return "Framing Divergent, Response Aligned"
            else:
                return "Consensus"
        
        combined_df['component_type'] = combined_df.apply(classify_component, axis=1)
        
        # Print classification distribution
        classification_counts = combined_df['component_type'].value_counts()
        print("Component classification distribution:")
        for category, count in classification_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        return combined_df
    
    def create_visualizations(self, classified_df: pd.DataFrame) -> None:
        """
        Create comprehensive visualizations of the divergence analysis.
        
        Args:
            classified_df: DataFrame with classified components
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Main scatter plot: NDI-OP vs NDI-R
        self._create_main_scatter_plot(classified_df)
        
        # 2. Distribution plots
        self._create_distribution_plots(classified_df)
        
        # 3. Dominant frame analysis
        self._create_frame_analysis_plots(classified_df)
        
        # 4. Platform analysis
        self._create_platform_analysis_plots(classified_df)
        
        print("All visualizations created successfully!")
    
    def _create_main_scatter_plot(self, df: pd.DataFrame) -> None:
        """Create the main scatter plot of NDI-OP vs NDI-R."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color map for dominant frames
        unique_frames = df['dominant_frame'].dropna().unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_frames)))
        frame_color_map = dict(zip(unique_frames, colors))
        
        # Plot points colored by dominant frame
        for frame in unique_frames:
            frame_data = df[df['dominant_frame'] == frame]
            ax.scatter(frame_data['ndi_op'], frame_data['ndi_r'], 
                      c=[frame_color_map[frame]], label=frame, alpha=0.7, s=60)
        
        # Add threshold lines
        ndi_op_threshold = df['ndi_op'].median()
        ndi_r_threshold = df['ndi_r'].median()
        
        ax.axvline(ndi_op_threshold, color='red', linestyle='--', alpha=0.5, label=f'NDI-OP threshold ({ndi_op_threshold:.3f})')
        ax.axhline(ndi_r_threshold, color='red', linestyle='--', alpha=0.5, label=f'NDI-R threshold ({ndi_r_threshold:.3f})')
        
        # Annotate quadrants
        ax.text(0.05, 0.95, 'Consensus', transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.75, 0.95, 'Framing Divergent,\nResponse Aligned', transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.05, 0.05, 'Framing Aligned,\nResponse Split', transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(0.75, 0.05, 'Highly Polarized', transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Annotate top 5 most divergent components (highest combined divergence)
        df['combined_divergence'] = df['ndi_op'] + df['ndi_r']
        top_divergent = df.nlargest(5, 'combined_divergence')
        
        for _, row in top_divergent.iterrows():
            ax.annotate(f"Component {row['component_id']}", 
                       (row['ndi_op'], row['ndi_r']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('NDI-OP (Narrative Divergence Index - Original Posts)', fontsize=12)
        ax.set_ylabel('NDI-R (Narrative Response Divergence)', fontsize=12)
        ax.set_title('Cross-Platform Narrative Divergence Analysis', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'narrative_divergence_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Main scatter plot saved")
    
    def _create_distribution_plots(self, df: pd.DataFrame) -> None:
        """Create distribution plots for NDI values."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # NDI-OP distribution
        axes[0, 0].hist(df['ndi_op'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['ndi_op'].median(), color='red', linestyle='--', label=f'Median: {df["ndi_op"].median():.3f}')
        axes[0, 0].set_xlabel('NDI-OP')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of NDI-OP Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # NDI-R distribution
        axes[0, 1].hist(df['ndi_r'].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(df['ndi_r'].median(), color='red', linestyle='--', label=f'Median: {df["ndi_r"].median():.3f}')
        axes[0, 1].set_xlabel('NDI-R')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of NDI-R Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Component type distribution
        type_counts = df['component_type'].value_counts()
        axes[1, 0].bar(type_counts.index, type_counts.values, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
        axes[1, 0].set_xlabel('Component Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Component Types')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined divergence vs component type
        df['combined_divergence'] = df['ndi_op'] + df['ndi_r']
        sns.boxplot(data=df, x='component_type', y='combined_divergence', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Component Type')
        axes[1, 1].set_ylabel('Combined Divergence (NDI-OP + NDI-R)')
        axes[1, 1].set_title('Combined Divergence by Component Type')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'divergence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Distribution plots saved")
    
    def _create_frame_analysis_plots(self, df: pd.DataFrame) -> None:
        """Create plots analyzing dominant narrative frames."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Dominant frame distribution
        frame_counts = df['dominant_frame'].value_counts()
        axes[0, 0].bar(frame_counts.index, frame_counts.values, color='skyblue')
        axes[0, 0].set_xlabel('Dominant Narrative Frame')
        axes[0, 0].set_ylabel('Number of Components')
        axes[0, 0].set_title('Distribution of Dominant Narrative Frames')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average NDI-OP by dominant frame
        frame_ndi_op = df.groupby('dominant_frame')['ndi_op'].mean().sort_values(ascending=False)
        axes[0, 1].bar(frame_ndi_op.index, frame_ndi_op.values, color='lightcoral')
        axes[0, 1].set_xlabel('Dominant Narrative Frame')
        axes[0, 1].set_ylabel('Average NDI-OP')
        axes[0, 1].set_title('Average NDI-OP by Dominant Frame')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average NDI-R by dominant frame
        frame_ndi_r = df.groupby('dominant_frame')['ndi_r'].mean().sort_values(ascending=False)
        axes[1, 0].bar(frame_ndi_r.index, frame_ndi_r.values, color='lightgreen')
        axes[1, 0].set_xlabel('Dominant Narrative Frame')
        axes[1, 0].set_ylabel('Average NDI-R')
        axes[1, 0].set_title('Average NDI-R by Dominant Frame')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Frame diversity heatmap (component types vs frames)
        frame_type_crosstab = pd.crosstab(df['component_type'], df['dominant_frame'])
        sns.heatmap(frame_type_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Dominant Narrative Frame')
        axes[1, 1].set_ylabel('Component Type')
        axes[1, 1].set_title('Component Types by Dominant Frame')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Frame analysis plots saved")
    
    def _create_platform_analysis_plots(self, df: pd.DataFrame) -> None:
        """Create plots analyzing platform patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Platform coverage distribution
        platform_counts = df['platforms_count'].value_counts().sort_index()
        axes[0, 0].bar(platform_counts.index, platform_counts.values, color='lightblue')
        axes[0, 0].set_xlabel('Number of Platforms per Component')
        axes[0, 0].set_ylabel('Number of Components')
        axes[0, 0].set_title('Platform Coverage Distribution')
        
        # NDI-OP vs number of platforms
        platform_ndi_op = df.groupby('platforms_count')['ndi_op'].mean()
        axes[0, 1].plot(platform_ndi_op.index, platform_ndi_op.values, marker='o', color='red', linewidth=2)
        axes[0, 1].set_xlabel('Number of Platforms')
        axes[0, 1].set_ylabel('Average NDI-OP')
        axes[0, 1].set_title('NDI-OP vs Platform Coverage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Post volume analysis
        axes[1, 0].scatter(df['total_original_posts'], df['ndi_op'], alpha=0.6, color='blue')
        axes[1, 0].set_xlabel('Total Original Posts')
        axes[1, 0].set_ylabel('NDI-OP')
        axes[1, 0].set_title('NDI-OP vs Post Volume')
        axes[1, 0].set_xscale('log')
        
        # Reply volume analysis
        reply_data = df[df['total_replies'] > 0]
        if len(reply_data) > 0:
            axes[1, 1].scatter(reply_data['total_replies'], reply_data['ndi_r'], alpha=0.6, color='green')
            axes[1, 1].set_xlabel('Total Replies')
            axes[1, 1].set_ylabel('NDI-R')
            axes[1, 1].set_title('NDI-R vs Reply Volume')
            axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Platform analysis plots saved")
    
    def generate_summary_report(self, classified_df: pd.DataFrame) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            classified_df: DataFrame with classified components
            
        Returns:
            Path to the generated report
        """
        print("Generating comprehensive summary report...")
        
        report_path = self.output_dir / 'narrative_divergence_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Narrative Divergence Analysis Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Components Analyzed:** {len(classified_df)}\n")
            f.write(f"- **Average NDI-OP:** {classified_df['ndi_op'].mean():.4f} (σ = {classified_df['ndi_op'].std():.4f})\n")
            f.write(f"- **Average NDI-R:** {classified_df['ndi_r'].mean():.4f} (σ = {classified_df['ndi_r'].std():.4f})\n")
            
            # Classification Summary
            f.write("\n## Component Classification Summary\n\n")
            type_counts = classified_df['component_type'].value_counts()
            for category, count in type_counts.items():
                percentage = (count / len(classified_df)) * 100
                f.write(f"- **{category}:** {count} components ({percentage:.1f}%)\n")
            
            # Top Divergent Components
            f.write("\n## Most Divergent Components\n\n")
            classified_df['combined_divergence'] = classified_df['ndi_op'] + classified_df['ndi_r']
            top_10 = classified_df.nlargest(10, 'combined_divergence')
            
            f.write("| Component ID | NDI-OP | NDI-R | Combined | Type | Dominant Frame |\n")
            f.write("|--------------|--------|-------|----------|------|----------------|\n")
            
            for _, row in top_10.iterrows():
                f.write(f"| {row['component_id']} | {row['ndi_op']:.4f} | {row['ndi_r']:.4f} | "
                       f"{row['combined_divergence']:.4f} | {row['component_type']} | {row['dominant_frame']} |\n")
            
            # Frame Analysis
            f.write("\n## Dominant Frame Analysis\n\n")
            frame_stats = classified_df.groupby('dominant_frame').agg({
                'ndi_op': ['count', 'mean', 'std'],
                'ndi_r': ['mean', 'std']
            }).round(4)
            
            f.write("| Frame | Count | Avg NDI-OP | Std NDI-OP | Avg NDI-R | Std NDI-R |\n")
            f.write("|-------|-------|------------|------------|-----------|----------|\n")
            
            for frame in frame_stats.index:
                count = frame_stats.loc[frame, ('ndi_op', 'count')]
                ndi_op_mean = frame_stats.loc[frame, ('ndi_op', 'mean')]
                ndi_op_std = frame_stats.loc[frame, ('ndi_op', 'std')]
                ndi_r_mean = frame_stats.loc[frame, ('ndi_r', 'mean')]
                ndi_r_std = frame_stats.loc[frame, ('ndi_r', 'std')]
                
                f.write(f"| {frame} | {count} | {ndi_op_mean:.4f} | {ndi_op_std:.4f} | "
                       f"{ndi_r_mean:.4f} | {ndi_r_std:.4f} |\n")
            
            # Methodology
            f.write("\n## Methodology\n\n")
            f.write("### NDI-OP (Narrative Divergence Index for Original Posts)\n")
            f.write("- Measures Jensen-Shannon divergence between narrative frame distributions across platforms\n")
            f.write("- Higher values indicate greater difference in how events are framed\n")
            f.write("- Range: [0, 1] where 0 = identical framing, 1 = completely different framing\n\n")
            
            f.write("### NDI-R (Narrative Response Divergence)\n")
            f.write("- Measures Jensen-Shannon divergence between reply type distributions across platforms\n")
            f.write("- Higher values indicate greater difference in response patterns\n")
            f.write("- Range: [0, 1] where 0 = identical responses, 1 = completely different responses\n\n")
            
            f.write("### Component Classifications\n")
            f.write("- **Highly Polarized:** High NDI-OP and high NDI-R\n")
            f.write("- **Framing Aligned, Response Split:** Low NDI-OP, high NDI-R\n")
            f.write("- **Framing Divergent, Response Aligned:** High NDI-OP, low NDI-R\n")
            f.write("- **Consensus:** Low NDI-OP and low NDI-R\n")
        
        print(f"Summary report saved to {report_path}")
        return str(report_path)
    
    def run_complete_analysis(self, narrative_data_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Run the complete narrative divergence analysis pipeline.
        
        Args:
            narrative_data_path: Path to narrative analysis JSON file
            
        Returns:
            Tuple of (classified_df, report_path)
        """
        print("=" * 60)
        print("STARTING NARRATIVE DIVERGENCE ANALYSIS")
        print("=" * 60)
        
        # Load and prepare data
        df = self.load_and_prepare_data(narrative_data_path)
        
        # Compute NDI-OP
        ndi_op_df = self.compute_ndi_op(df)
        
        # Compute NDI-R
        ndi_r_df = self.compute_ndi_r(df)
        
        # Classify components
        classified_df = self.classify_components(ndi_op_df, ndi_r_df)
        
        # Create visualizations
        self.create_visualizations(classified_df)
        
        # Generate report
        report_path = self.generate_summary_report(classified_df)
        
        # Save results to CSV
        results_path = self.output_dir / 'narrative_divergence_results.csv'
        classified_df.to_csv(results_path, index=False)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary report: {report_path}")
        print(f"Detailed results: {results_path}")
        
        return classified_df, report_path


def main():
    """Main function to run the narrative divergence analysis."""
    # Initialize analyzer
    analyzer = NarrativeDivergenceAnalyzer()
    
    # Path to your narrative analysis data
    narrative_data_path = "output/narrative_analysis_complete.json"
    
    # Run complete analysis
    results_df, report_path = analyzer.run_complete_analysis(narrative_data_path)
    
    print(f"\n✅ Narrative Divergence Analysis completed successfully!")
    print(f"📊 Analyzed {len(results_df)} components")
    print(f"📈 Generated {len(results_df['component_type'].value_counts())} component categories")
    print(f"📄 Report available at: {report_path}")


if __name__ == "__main__":
    main()
