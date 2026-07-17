#!/usr/bin/env python3
"""
Comprehensive Narrative Analysis Results Analyzer

This script analyzes the narrative analysis results from cross-platform analysis,
providing insights into narrative frame distributions, platform differences,
and temporal patterns.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NarrativeResultsAnalyzer:
    """
    Comprehensive analyzer for narrative analysis results.
    """
    
    def __init__(self, output_dir: str = "output", analysis_output_dir: str = "narrative_analysis_output"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory containing narrative analysis results
            analysis_output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.analysis_output_dir = Path(analysis_output_dir)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.narrative_data = []
        self.summary_stats = {}
        self.platform_stats = {}
        self.frame_stats = {}
        
        # Narrative frames (update based on your actual frames)
        self.narrative_frames = [
            "Persecution", "Corruption", "Accountability", "Irony/Detachment",
            "Heroism", "Civic Critique", "Moral Decay", "Media Manipulation",
            "Strategic Pragmatism", "Cultural Identity"
        ]
        
        logger.info(f"Initialized NarrativeResultsAnalyzer")
        logger.info(f"Input directory: {self.output_dir}")
        logger.info(f"Analysis output directory: {self.analysis_output_dir}")
    
    def load_narrative_data(self, sample_size: Optional[int] = None) -> bool:
        """
        Load narrative analysis data from JSON files.
        
        Args:
            sample_size: If provided, only load first N components for faster analysis
        
        Returns:
            True if data loaded successfully
        """
        logger.info("Loading narrative analysis data...")
        
        # Try to load complete data first
        complete_file = self.output_dir / "narrative_analysis_complete.json"
        
        if complete_file.exists():
            logger.info(f"Loading complete narrative analysis from {complete_file}")
            try:
                with open(complete_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Handle both list format and dict format
                if isinstance(data, list):
                    self.narrative_data = data
                else:
                    self.narrative_data = data.get('results', [])
                logger.info(f"Loaded {len(self.narrative_data)} complete narrative analyses")
                
                # Apply sampling if requested
                if sample_size and len(self.narrative_data) > sample_size:
                    self.narrative_data = self.narrative_data[:sample_size]
                    logger.info(f"Sampled down to {len(self.narrative_data)} analyses")
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load complete file: {e}")
        
        # Fall back to partial files
        logger.info("Loading from partial files...")
        partial_files = sorted([f for f in self.output_dir.glob("narrative_analysis_partial_*.json")])
        
        if not partial_files:
            logger.error("No narrative analysis files found")
            return False
        
        logger.info(f"Found {len(partial_files)} partial files")
        
        loaded_count = 0
        for file_path in partial_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'results' in data:
                    self.narrative_data.extend(data['results'])
                    loaded_count += len(data['results'])
                    
                    # Stop if we've reached the sample size
                    if sample_size and loaded_count >= sample_size:
                        self.narrative_data = self.narrative_data[:sample_size]
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.narrative_data)} narrative analyses from {len(partial_files)} files")
        return len(self.narrative_data) > 0
    
    def analyze_narrative_frames(self) -> Dict[str, Any]:
        """
        Analyze narrative frame distributions across all data.
        
        Returns:
            Dictionary with frame analysis results
        """
        logger.info("Analyzing narrative frame distributions...")
        
        frame_scores = defaultdict(list)
        frame_counts = defaultdict(int)
        platform_frame_scores = defaultdict(lambda: defaultdict(list))
        
        for component in self.narrative_data:
            component_id = component.get('component_id', 'unknown')
            narrative_frames = component.get('narrative_frames', {})
            
            # Process each platform's narrative frames
            for platform_name, frames_list in narrative_frames.items():
                if isinstance(frames_list, list):
                    for frame_name in frames_list:
                        if frame_name:  # Make sure frame name is not empty
                            frame_counts[frame_name] += 1
                            platform_frame_scores[platform_name][frame_name].append(1.0)  # Binary presence
        
        # Calculate statistics
        results = {
            'frame_statistics': {},
            'platform_differences': {},
            'frame_rankings': {}
        }
        
        # Calculate total posts for rate calculation
        total_posts = len(self.narrative_data)
        
        # Overall frame statistics
        for frame_name, count in frame_counts.items():
            results['frame_statistics'][frame_name] = {
                'total_count': count,
                'presence_rate': count / total_posts if total_posts > 0 else 0
            }
        
        # Platform-specific statistics
        platform_totals = {}
        for platform_name, platform_frames_dict in platform_frame_scores.items():
            platform_total = sum(len(frame_scores) for frame_scores in platform_frames_dict.values())
            platform_totals[platform_name] = platform_total
            
            results['platform_differences'][platform_name] = {}
            for frame_name, scores in platform_frames_dict.items():
                if scores:
                    results['platform_differences'][platform_name][frame_name] = {
                        'count': len(scores),
                        'platform_presence_rate': len(scores) / platform_total if platform_total > 0 else 0
                    }
        
        # Frame rankings by prevalence
        frame_prevalence = [(frame, stats['presence_rate']) 
                           for frame, stats in results['frame_statistics'].items()]
        results['frame_rankings']['by_prevalence'] = sorted(frame_prevalence, 
                                                           key=lambda x: x[1], reverse=True)
        
        # Frame rankings by total count
        frame_counts_sorted = [(frame, stats['total_count']) 
                              for frame, stats in results['frame_statistics'].items()]
        results['frame_rankings']['by_total_count'] = sorted(frame_counts_sorted, 
                                                            key=lambda x: x[1], reverse=True)
        
        self.frame_stats = results
        logger.info(f"Analyzed {len(frame_counts)} different narrative frames from {total_posts} components")
        return results
    
    def analyze_platform_differences(self) -> Dict[str, Any]:
        """
        Analyze differences in narrative frames across platforms.
        
        Returns:
            Dictionary with platform comparison results
        """
        logger.info("Analyzing platform differences...")
        
        platform_data = defaultdict(lambda: defaultdict(int))
        platform_post_counts = defaultdict(int)
        
        for component in self.narrative_data:
            narrative_frames = component.get('narrative_frames', {})
            platform_counts = component.get('platform_counts', {})
            
            # Use platform_counts for post counts
            for platform_name, count in platform_counts.items():
                platform_post_counts[platform_name] += count
            
            # Count frame occurrences per platform
            for platform_name, frames_list in narrative_frames.items():
                if isinstance(frames_list, list):
                    for frame_name in frames_list:
                        if frame_name:
                            platform_data[platform_name][frame_name] += 1
        
        results = {
            'platform_summary': {},
            'frame_differences': {},
            'platform_rankings': {}
        }
        
        # Platform summary
        for platform_name, frame_data in platform_data.items():
            total_frame_count = sum(frame_data.values())
            frame_rates = {}
            
            for frame_name, count in frame_data.items():
                if total_frame_count > 0:
                    frame_rates[frame_name] = count / total_frame_count
            
            results['platform_summary'][platform_name] = {
                'total_posts_analyzed': platform_post_counts.get(platform_name, 0),
                'total_frame_instances': total_frame_count,
                'average_frame_score': sum(frame_rates.values()) / len(frame_rates) if frame_rates else 0,
                'dominant_frames': sorted(frame_rates.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        # Frame differences across platforms
        for frame_name in self.narrative_frames:
            platform_stats = {}
            for platform_name, frame_data in platform_data.items():
                if frame_name in frame_data:
                    total_platform_frames = sum(frame_data.values())
                    platform_stats[platform_name] = {
                        'count': frame_data[frame_name],
                        'rate': frame_data[frame_name] / total_platform_frames if total_platform_frames > 0 else 0,
                        'total_platform_frames': total_platform_frames
                    }
            
            if len(platform_stats) > 1:
                results['frame_differences'][frame_name] = platform_stats
        
        self.platform_stats = results
        logger.info(f"Analyzed {len(platform_data)} platforms")
        return results
    
    def create_visualizations(self) -> bool:
        """
        Create comprehensive visualizations of the narrative analysis.
        
        Returns:
            True if visualizations created successfully
        """
        logger.info("Creating narrative analysis visualizations...")
        
        try:
            # Set up matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Frame prevalence chart
            self._create_frame_prevalence_chart()
            
            # 2. Platform comparison heatmap
            self._create_platform_heatmap()
            
            # 3. Frame score distributions
            self._create_frame_distributions()
            
            # 4. Interactive plotly dashboard
            self._create_interactive_dashboard()
            
            # 5. Summary statistics table
            self._create_summary_table()
            
            logger.info("All visualizations created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return False
    
    def _create_frame_prevalence_chart(self):
        """Create a chart showing narrative frame prevalence."""
        if not self.frame_stats:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prevalence rates
        frames = [item[0] for item in self.frame_stats['frame_rankings']['by_prevalence']]
        prevalence = [item[1] for item in self.frame_stats['frame_rankings']['by_prevalence']]
        
        bars1 = ax1.barh(frames, prevalence, color='skyblue')
        ax1.set_xlabel('Prevalence Rate')
        ax1.set_title('Narrative Frame Prevalence Rates')
        ax1.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, prevalence)):
            ax1.text(val + 0.01, i, f'{val:.3f}', va='center')
        
        # Total counts
        frames_count = [item[0] for item in self.frame_stats['frame_rankings']['by_total_count']]
        counts_total = [item[1] for item in self.frame_stats['frame_rankings']['by_total_count']]
        
        bars2 = ax2.barh(frames_count, counts_total, color='lightcoral')
        ax2.set_xlabel('Total Count')
        ax2.set_title('Total Narrative Frame Counts')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, counts_total)):
            ax2.text(val + 0.01, i, f'{val}', va='center')
        
        plt.tight_layout()
        plt.savefig(self.analysis_output_dir / 'frame_prevalence_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Frame prevalence chart saved")
    
    def _create_platform_heatmap(self):
        """Create a heatmap comparing frames across platforms."""
        if not self.platform_stats:
            return
        
        # Prepare data for heatmap
        platforms = list(self.platform_stats['platform_summary'].keys())
        
        # Create matrix of average scores
        heatmap_data = []
        frame_names = []
        
        for frame_name in self.narrative_frames:
            if frame_name in self.frame_stats.get('frame_differences', {}):
                frame_scores = []
                for platform in platforms:
                    platform_data = self.platform_stats['platform_summary'][platform]
                    # Find frame score in dominant frames
                    frame_score = 0
                    for fname, score in platform_data['dominant_frames']:
                        if fname == frame_name:
                            frame_score = score
                            break
                    frame_scores.append(frame_score)
                
                if any(score > 0 for score in frame_scores):
                    heatmap_data.append(frame_scores)
                    frame_names.append(frame_name)
        
        if heatmap_data:
            # Create heatmap
            plt.figure(figsize=(10, 8))
            heatmap_array = np.array(heatmap_data)
            
            sns.heatmap(heatmap_array, 
                       xticklabels=platforms,
                       yticklabels=frame_names,
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Average Frame Score'})
            
            plt.title('Narrative Frame Intensity Across Platforms')
            plt.xlabel('Platform')
            plt.ylabel('Narrative Frame')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig(self.analysis_output_dir / 'platform_frame_heatmap.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Platform heatmap saved")
    
    def _create_frame_distributions(self):
        """Create distribution plots for frame scores."""
        if not self.frame_stats:
            return
        
        # Collect frame counts for histogram
        all_frame_data = defaultdict(list)
        
        for component in self.narrative_data[:1000]:  # Sample for performance
            narrative_frames = component.get('narrative_frames', {})
            for platform_name, frames_list in narrative_frames.items():
                if isinstance(frames_list, list):
                    for frame_name in frames_list:
                        if frame_name:
                            all_frame_data[frame_name].append(1.0)  # Binary presence
        
        # Create distribution plots
        n_frames = len(all_frame_data)
        if n_frames == 0:
            return
            
        cols = 3
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, (frame_name, scores) in enumerate(all_frame_data.items()):
            row, col = i // cols, i % cols
            
            if scores:
                axes[row][col].hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[row][col].set_title(f'{frame_name}\n(μ={np.mean(scores):.3f}, σ={np.std(scores):.3f})')
                axes[row][col].set_xlabel('Score')
                axes[row][col].set_ylabel('Frequency')
                axes[row][col].axvline(np.mean(scores), color='red', linestyle='--', alpha=0.7)
        
        # Hide empty subplots
        for i in range(n_frames, rows * cols):
            row, col = i // cols, i % cols
            axes[row][col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.analysis_output_dir / 'frame_score_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Frame distributions saved")
    
    def _create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard."""
        if not self.frame_stats or not self.platform_stats:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Frame Prevalence', 'Platform Comparison', 
                          'Score Distribution', 'Top Frames by Platform'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # 1. Frame prevalence
        frames = [item[0] for item in self.frame_stats['frame_rankings']['by_prevalence']]
        prevalence = [item[1] for item in self.frame_stats['frame_rankings']['by_prevalence']]
        
        fig.add_trace(
            go.Bar(x=prevalence, y=frames, orientation='h', name='Prevalence'),
            row=1, col=1
        )
        
        # 2. Platform comparison (simplified)
        platforms = list(self.platform_stats['platform_summary'].keys())
        avg_scores = [self.platform_stats['platform_summary'][p]['average_frame_score'] 
                     for p in platforms]
        
        fig.add_trace(
            go.Bar(x=platforms, y=avg_scores, name='Avg Score'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Narrative Analysis Dashboard",
            showlegend=False
        )
        
        # Save interactive plot
        fig.write_html(self.analysis_output_dir / 'interactive_dashboard.html')
        logger.info("Interactive dashboard saved")
    
    def _create_summary_table(self):
        """Create a summary statistics table."""
        if not self.frame_stats or not self.platform_stats:
            return
        
        # Prepare summary data
        summary_data = []
        
        for frame_name, stats in self.frame_stats['frame_statistics'].items():
            summary_data.append({
                'Frame': frame_name,
                'Total Count': stats['total_count'],
                'Presence Rate': f"{stats['presence_rate']:.3f}",
                'Components with Frame': stats['total_count']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Presence Rate', ascending=False)
        
        # Save to CSV
        df_summary.to_csv(self.analysis_output_dir / 'narrative_frame_summary.csv', index=False)
        
        # Platform summary
        platform_summary = []
        for platform, stats in self.platform_stats['platform_summary'].items():
            platform_summary.append({
                'Platform': platform,
                'Posts Analyzed': stats['total_posts_analyzed'],
                'Frame Instances': stats['total_frame_instances'],
                'Average Frame Rate': f"{stats['average_frame_score']:.3f}",
                'Top Frame': stats['dominant_frames'][0][0] if stats['dominant_frames'] else 'N/A',
                'Top Frame Rate': f"{stats['dominant_frames'][0][1]:.3f}" if stats['dominant_frames'] else 'N/A'
            })
        
        df_platforms = pd.DataFrame(platform_summary)
        df_platforms.to_csv(self.analysis_output_dir / 'platform_summary.csv', index=False)
        
        logger.info("Summary tables saved")
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive text report of the analysis.
        
        Returns:
            Path to the generated report file
        """
        logger.info("Generating comprehensive report...")
        
        report_path = self.analysis_output_dir / 'narrative_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Narrative Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Components Analyzed**: {len(self.narrative_data)}\n")
            
            if self.frame_stats:
                total_frames = len(self.frame_stats['frame_statistics'])
                f.write(f"- **Narrative Frames Identified**: {total_frames}\n")
                
                # Most prevalent frame
                if self.frame_stats['frame_rankings']['by_prevalence']:
                    top_frame, top_prevalence = self.frame_stats['frame_rankings']['by_prevalence'][0]
                    f.write(f"- **Most Prevalent Frame**: {top_frame} ({top_prevalence:.3f})\n")
            
            if self.platform_stats:
                platforms = list(self.platform_stats['platform_summary'].keys())
                f.write(f"- **Platforms Analyzed**: {', '.join(platforms)}\n\n")
            
            # Frame Analysis
            if self.frame_stats:
                f.write("## Narrative Frame Analysis\n\n")
                f.write("### Frame Rankings by Prevalence\n\n")
                
                for i, (frame, prevalence) in enumerate(self.frame_stats['frame_rankings']['by_prevalence'][:10], 1):
                    f.write(f"{i}. **{frame}**: {prevalence:.3f} prevalence rate\n")
                
                f.write("\n### Frame Statistics\n\n")
                f.write("| Frame | Total Count | Prevalence Rate | Components |\n")
                f.write("|-------|-------------|-----------------|------------|\n")
                
                for frame, stats in sorted(self.frame_stats['frame_statistics'].items(), 
                                         key=lambda x: x[1]['presence_rate'], reverse=True):
                    f.write(f"| {frame} | {stats['total_count']} | {stats['presence_rate']:.3f} | {len(self.narrative_data)} |\n")
            
            # Platform Analysis
            if self.platform_stats:
                f.write("\n## Platform Analysis\n\n")
                
                for platform, stats in self.platform_stats['platform_summary'].items():
                    f.write(f"### {platform}\n\n")
                    f.write(f"- **Posts Analyzed**: {stats['total_posts_analyzed']}\n")
                    f.write(f"- **Frame Instances**: {stats['total_frame_instances']}\n")
                    f.write(f"- **Average Frame Rate**: {stats['average_frame_score']:.3f}\n")
                    f.write("- **Top 5 Frames**:\n")
                    
                    for i, (frame, score) in enumerate(stats['dominant_frames'][:5], 1):
                        f.write(f"  {i}. {frame}: {score:.3f}\n")
                    f.write("\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write("This analysis was performed on narrative classification results from cross-platform ")
            f.write("social media data. The narrative frames were identified using advanced NLP models ")
            f.write("and represent different rhetorical and thematic patterns in the discourse.\n\n")
            
            f.write("### Frame Definitions\n\n")
            f.write("The analysis uses the following narrative frames:\n\n")
            for frame in self.narrative_frames:
                f.write(f"- **{frame}**\n")
            
            f.write("\n### Analysis Metrics\n\n")
            f.write("- **Prevalence Rate**: Proportion of components containing each frame\n")
            f.write("- **Total Count**: Total occurrences of each frame across all components\n")
            f.write("- **Platform Comparison**: Frame frequency differences across platforms\n")
        
        logger.info(f"Comprehensive report saved to {report_path}")
        return str(report_path)
    
    def run_complete_analysis(self, sample_size: Optional[int] = None) -> bool:
        """
        Run the complete narrative analysis pipeline.
        
        Args:
            sample_size: Optional limit on number of components to analyze
        
        Returns:
            True if analysis completed successfully
        """
        logger.info("=== Starting Comprehensive Narrative Analysis ===")
        
        try:
            # Load data
            if not self.load_narrative_data(sample_size=sample_size):
                logger.error("Failed to load narrative data")
                return False
            
            # Analyze frames
            self.analyze_narrative_frames()
            
            # Analyze platforms
            self.analyze_platform_differences()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            report_path = self.generate_comprehensive_report()
            
            logger.info("=== Analysis Complete ===")
            logger.info(f"Results saved to: {self.analysis_output_dir}")
            logger.info(f"Report available at: {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("NARRATIVE ANALYSIS SUMMARY")
            print("="*60)
            print(f"Components analyzed: {len(self.narrative_data)}")
            
            if self.frame_stats:
                print(f"Narrative frames found: {len(self.frame_stats['frame_statistics'])}")
                if self.frame_stats['frame_rankings']['by_prevalence']:
                    top_frame, top_prevalence = self.frame_stats['frame_rankings']['by_prevalence'][0]
                    print(f"Most prevalent frame: {top_frame} ({top_prevalence:.3f})")
            
            if self.platform_stats:
                platforms = list(self.platform_stats['platform_summary'].keys())
                print(f"Platforms analyzed: {', '.join(platforms)}")
            
            print(f"\nResults directory: {self.analysis_output_dir}")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run narrative analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze narrative analysis results')
    parser.add_argument('--output-dir', default='output', 
                       help='Directory containing narrative analysis results')
    parser.add_argument('--analysis-output-dir', default='narrative_analysis_output',
                       help='Directory to save analysis outputs')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit analysis to first N components (for faster processing)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = NarrativeResultsAnalyzer(
        output_dir=args.output_dir,
        analysis_output_dir=args.analysis_output_dir
    )
    
    # Run analysis
    success = analyzer.run_complete_analysis(sample_size=args.sample_size)
    
    if success:
        print("\n✅ Narrative analysis completed successfully!")
    else:
        print("\n❌ Narrative analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
