#!/usr/bin/env python3
"""
Unified Platform Narratives Analysis Pipeline

This script combines cross-platform analysis and reply analysis into a single
configurable pipeline that allows users to choose which steps to execute.

Author: Data Analyst
Date: June 13, 2025
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure the repo root is on the path so `src` imports as a package
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis modules
from src.cross_platform_analyzer import CrossPlatformAnalyzer
from src.reply_analyzer import ReplyAnalyzer
from src.config_manager import ConfigManager
from src.utils import setup_logging, save_pipeline_state, load_pipeline_state

# Set up logging
logger = logging.getLogger(__name__)


class UnifiedPlatformAnalyzer:
    """
    Unified analyzer that combines cross-platform analysis and reply analysis
    with configurable pipeline steps.
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "unified_analysis_output"):
        """
        Initialize the unified analyzer.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory to save all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize analyzers
        self.cross_platform_analyzer = None
        self.reply_analyzer = None
        
        # Pipeline state tracking
        self.pipeline_state = {
            'completed_steps': [],
            'current_step': None,
            'start_time': None,
            'step_times': {},
            'data_paths': {}
        }
        
        logger.info(f"Initialized UnifiedPlatformAnalyzer")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_analyzers(self) -> bool:
        """Setup the individual analyzers."""
        logger.info("Setting up analyzers...")
        
        try:
            # Setup cross-platform analyzer
            self.cross_platform_analyzer = CrossPlatformAnalyzer(
                config=self.config,
                output_dir=str(self.output_dir / "cross_platform")
            )
            
            # Setup reply analyzer
            self.reply_analyzer = ReplyAnalyzer(
                output_dir=str(self.output_dir / "reply_analysis")
            )
            
            logger.info("✅ Analyzers setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup analyzers: {e}")
            return False
    
    def run_pipeline(self, steps: List[str], resume_from: Optional[str] = None) -> bool:
        """
        Run the complete analysis pipeline with selected steps.
        
        Args:
            steps: List of pipeline steps to execute
            resume_from: Step to resume from (if resuming a previous run)
            
        Returns:
            Success status
        """
        logger.info("Starting unified platform narratives analysis pipeline")
        logger.info(f"Selected steps: {steps}")
        logger.info("=" * 70)
        
        self.pipeline_state['start_time'] = time.time()
        
        # Define all available pipeline steps
        available_steps = {
            'setup': self.step_setup,
            'load_data': self.step_load_data,
            'cross_platform_matching': self.step_cross_platform_matching,
            'narrative_analysis': self.step_narrative_analysis,
            'conversation_graphs': self.step_conversation_graphs,
            'reply_classification': self.step_reply_classification,
            'combined_analysis': self.step_combined_analysis,
            'generate_reports': self.step_generate_reports
        }
        
        # Validate steps
        invalid_steps = [step for step in steps if step not in available_steps]
        if invalid_steps:
            logger.error(f"Invalid steps: {invalid_steps}")
            logger.error(f"Available steps: {list(available_steps.keys())}")
            return False
        
        # Load previous state if resuming
        if resume_from:
            logger.info(f"Resuming from step: {resume_from}")
            if not self.load_previous_state():
                logger.warning("Could not load previous state, starting fresh")
        
        # Execute steps
        success = True
        for step_name in steps:
            if resume_from and step_name != resume_from and resume_from not in self.pipeline_state['completed_steps']:
                logger.info(f"Skipping {step_name} (resuming from {resume_from})")
                continue
            
            if step_name in self.pipeline_state['completed_steps']:
                logger.info(f"Step {step_name} already completed, skipping")
                continue
            
            logger.info(f"\n{'='*20} STEP: {step_name.upper()} {'='*20}")
            step_start = time.time()
            self.pipeline_state['current_step'] = step_name
            
            try:
                step_success = available_steps[step_name]()
                step_time = time.time() - step_start
                self.pipeline_state['step_times'][step_name] = step_time
                
                if step_success:
                    self.pipeline_state['completed_steps'].append(step_name)
                    logger.info(f"✅ Step {step_name} completed in {step_time:.2f}s")
                    
                    # Save state after each successful step
                    self.save_current_state()
                else:
                    logger.error(f"❌ Step {step_name} failed")
                    success = False
                    break
                    
            except Exception as e:
                logger.error(f"❌ Step {step_name} failed with error: {e}")
                traceback.print_exc()
                success = False
                break
        
        # Final summary
        total_time = time.time() - self.pipeline_state['start_time']
        self.pipeline_state['total_time'] = total_time
        
        logger.info("\n" + "="*70)
        if success:
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("💥 Pipeline failed")
        
        logger.info(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f}m)")
        logger.info(f"📁 Results saved in: {self.output_dir}")
        
        # Save final state
        self.save_current_state()
        
        return success
    
    def step_setup(self) -> bool:
        """Setup step: Initialize analyzers and vLLM if available."""
        logger.info("Setting up analyzers and checking vLLM availability...")
        
        if not self.setup_analyzers():
            return False
        
        # Try to setup vLLM for both analyzers
        vllm_success_cross = self.cross_platform_analyzer.setup_vllm_pipeline()
        vllm_success_reply = self.reply_analyzer.setup_vllm_pipeline()
        
        logger.info(f"vLLM setup - Cross-platform: {vllm_success_cross}, Reply: {vllm_success_reply}")
        
        return True
    
    def step_load_data(self) -> bool:
        """Load and preprocess data from all platforms."""
        logger.info("Loading and preprocessing platform data...")
        
        # Load data using cross-platform analyzer
        if not self.cross_platform_analyzer.load_platform_data():
            logger.error("Failed to load platform data")
            return False
        
        # Extract and preprocess posts
        if not self.cross_platform_analyzer.extract_posts_from_platform_data():
            logger.error("Failed to preprocess posts")
            return False
        
        # Create summary statistics
        summary_df = self.cross_platform_analyzer.create_summary_statistics()
        
        # Store data paths for later steps
        self.pipeline_state['data_paths']['processed_data'] = str(self.output_dir / "cross_platform" / "processed_posts.json")
        self.pipeline_state['data_paths']['summary'] = str(self.output_dir / "cross_platform" / "platform_summary.csv")
        
        return True
    
    def step_cross_platform_matching(self) -> bool:
        """Build similarity graphs and extract matched components."""
        logger.info("Building cross-platform similarity graphs...")
        
        # Build similarity graph
        if not self.cross_platform_analyzer.build_similarity_graph():
            logger.error("Failed to build similarity graph")
            return False
        
        # Extract matched components
        if not self.cross_platform_analyzer.extract_matched_components():
            logger.error("Failed to extract matched components")
            return False
        
        # Store matched components path
        self.pipeline_state['data_paths']['matched_components'] = str(self.output_dir / "cross_platform" / "matched_components.json")
        
        return True
    
    def step_narrative_analysis(self) -> bool:
        """Perform narrative frame analysis on matched components."""
        logger.info("Analyzing narrative frames...")
        
        # Run narrative analysis
        if not self.cross_platform_analyzer.analyze_narrative_frames():
            logger.error("Failed to analyze narrative frames")
            return False
        
        # Generate visualizations
        if not self.cross_platform_analyzer.create_narrative_visualizations():
            logger.error("Failed to create narrative visualizations")
            return False
        
        return True
    
    def step_conversation_graphs(self) -> bool:
        """Build conversation graphs for reply analysis."""
        logger.info("Building conversation graphs...")
        
        # Load conversation data
        data_dir = self.config.get('data', {}).get('data_dir', 'data/data')
        conversation_data = self.reply_analyzer.load_conversation_data(data_dir)
        
        if not conversation_data:
            logger.error("Failed to load conversation data")
            return False
        
        # Build conversation graphs
        conversation_graphs = self.reply_analyzer.build_conversation_graphs(conversation_data)
        
        if not conversation_graphs:
            logger.error("Failed to build conversation graphs")
            return False
        
        # Store conversation data
        self.pipeline_state['data_paths']['conversation_graphs'] = str(self.output_dir / "reply_analysis" / "conversation_graphs.json")
        
        return True
    
    def step_reply_classification(self) -> bool:
        """Identify and classify replies to root posts."""
        logger.info("Classifying replies to root posts...")
        
        # Load matched components
        matched_components_path = self.pipeline_state['data_paths'].get('matched_components')
        if not matched_components_path or not Path(matched_components_path).exists():
            logger.error("Matched components not found. Run cross_platform_matching step first.")
            return False
        
        # Load matched components
        matched_components = self.reply_analyzer.load_matched_components(matched_components_path)
        
        # Load conversation data
        data_dir = self.config.get('data', {}).get('data_dir', 'data/data')
        conversation_data = self.reply_analyzer.load_conversation_data(data_dir)
        conversation_graphs = self.reply_analyzer.build_conversation_graphs(conversation_data)
        
        # Identify root posts and replies
        components_with_replies = self.reply_analyzer.identify_root_posts_in_components(
            matched_components, conversation_graphs
        )
        
        # Classify replies
        classified_components = self.reply_analyzer.classify_replies_in_components(components_with_replies)
        
        # Store results
        classified_path = self.output_dir / "reply_analysis" / "classified_replies.json"
        with open(classified_path, 'w', encoding='utf-8') as f:
            json.dump(classified_components, f, indent=2, ensure_ascii=False)
        
        self.pipeline_state['data_paths']['classified_replies'] = str(classified_path)
        
        return True
    
    def step_combined_analysis(self) -> bool:
        """Combine cross-platform and reply analyses."""
        logger.info("Performing combined analysis...")
        
        # Load results from both analyses
        classified_replies_path = self.pipeline_state['data_paths'].get('classified_replies')
        if not classified_replies_path or not Path(classified_replies_path).exists():
            logger.error("Classified replies not found. Run reply_classification step first.")
            return False
        
        # Perform combined analysis
        combined_results = self.analyze_cross_platform_reply_patterns()
        
        # Save combined results
        combined_path = self.output_dir / "combined_analysis_results.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        self.pipeline_state['data_paths']['combined_results'] = str(combined_path)
        
        return True
    
    def step_generate_reports(self) -> bool:
        """Generate final reports and visualizations."""
        logger.info("Generating final reports...")
        
        # Generate comprehensive report
        report_content = self.generate_comprehensive_report()
        
        # Save report
        report_path = self.output_dir / "comprehensive_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        return True
    
    def analyze_cross_platform_reply_patterns(self) -> Dict[str, Any]:
        """Analyze patterns between cross-platform matches and reply behaviors."""
        logger.info("Analyzing cross-platform reply patterns...")
        
        # Load classified replies
        with open(self.pipeline_state['data_paths']['classified_replies'], 'r', encoding='utf-8') as f:
            classified_components = json.load(f)
        
        # Analyze patterns
        analysis_results = {
            'total_components_analyzed': len(classified_components),
            'platform_reply_patterns': {},
            'cross_platform_differences': {},
            'narrative_reply_correlations': {},
            'summary_statistics': {}
        }
        
        # Platform-specific reply patterns
        for platform in ['truth', 'bluesky', 'mastodon']:
            platform_stats = {
                'total_replies': 0,
                'reinforce_count': 0,
                'challenge_count': 0,
                'shift_count': 0
            }
            
            for component in classified_components:
                if platform in component.get('platforms', {}):
                    platform_data = component['platforms'][platform]
                    for root_post in platform_data.get('root_posts', []):
                        for reply in root_post.get('replies', []):
                            platform_stats['total_replies'] += 1
                            classification = reply.get('classification', {}).get('classification', 'shift')
                            platform_stats[f'{classification}_count'] += 1
            
            # Calculate percentages
            total = platform_stats['total_replies']
            if total > 0:
                platform_stats['reinforce_pct'] = (platform_stats['reinforce_count'] / total) * 100
                platform_stats['challenge_pct'] = (platform_stats['challenge_count'] / total) * 100
                platform_stats['shift_pct'] = (platform_stats['shift_count'] / total) * 100
            
            analysis_results['platform_reply_patterns'][platform] = platform_stats
        
        return analysis_results
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        logger.info("Generating comprehensive report...")
        
        report_lines = [
            "# Comprehensive Platform Narratives Analysis Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Pipeline Execution Summary",
            f"- Total execution time: {self.pipeline_state.get('total_time', 0):.2f} seconds",
            f"- Completed steps: {', '.join(self.pipeline_state['completed_steps'])}",
            "",
        ]
        
        # Add step timing details
        if self.pipeline_state['step_times']:
            report_lines.extend([
                "### Step Execution Times",
                ""
            ])
            for step, duration in self.pipeline_state['step_times'].items():
                report_lines.append(f"- {step}: {duration:.2f}s")
            report_lines.append("")
        
        # Add data summary
        if 'summary' in self.pipeline_state['data_paths']:
            report_lines.extend([
                "## Data Summary",
                "Platform data statistics can be found in the generated CSV files.",
                ""
            ])
        
        # Add file references
        report_lines.extend([
            "## Generated Files",
            f"- Output directory: `{self.output_dir}`",
            f"- Cross-platform analysis: `{self.output_dir}/cross_platform/`",
            f"- Reply analysis: `{self.output_dir}/reply_analysis/`",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def save_current_state(self):
        """Save current pipeline state."""
        state_path = self.output_dir / "pipeline_state.json"
        save_pipeline_state(self.pipeline_state, state_path)
    
    def load_previous_state(self) -> bool:
        """Load previous pipeline state."""
        state_path = self.output_dir / "pipeline_state.json"
        if state_path.exists():
            self.pipeline_state = load_pipeline_state(state_path)
            return True
        return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Platform Narratives Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  setup                 - Initialize analyzers and check vLLM availability
  load_data            - Load and preprocess platform data
  cross_platform_matching - Build similarity graphs and extract matched components
  narrative_analysis   - Perform narrative frame analysis
  conversation_graphs  - Build conversation graphs for reply analysis
  reply_classification - Classify replies to root posts
  combined_analysis    - Combine cross-platform and reply analyses
  generate_reports     - Generate final reports and visualizations

Examples:
  # Run full pipeline
  python unified_analysis.py --steps all
  
  # Run specific steps
  python unified_analysis.py --steps setup load_data cross_platform_matching
  
  # Resume from a specific step
  python unified_analysis.py --steps all --resume-from narrative_analysis
  
  # Use custom configuration
  python unified_analysis.py --config my_config.yaml --steps all
        """
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=[
            'setup', 'load_data', 'cross_platform_matching', 'narrative_analysis',
            'conversation_graphs', 'reply_classification', 'combined_analysis',
            'generate_reports', 'all'
        ],
        default=['all'],
        help='Pipeline steps to execute (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='unified_analysis_output',
        help='Output directory for results (default: unified_analysis_output)'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        choices=[
            'setup', 'load_data', 'cross_platform_matching', 'narrative_analysis',
            'conversation_graphs', 'reply_classification', 'combined_analysis',
            'generate_reports'
        ],
        help='Resume pipeline from this step'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--max-components',
        type=int,
        help='Maximum number of components to process (for testing)'
    )
    
    return parser


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    
    # Handle 'all' steps
    if 'all' in args.steps:
        args.steps = [
            'setup', 'load_data', 'cross_platform_matching', 'narrative_analysis',
            'conversation_graphs', 'reply_classification', 'combined_analysis',
            'generate_reports'
        ]
    
    logger.info("Starting Unified Platform Narratives Analysis")
    logger.info(f"Selected steps: {args.steps}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize analyzer
        analyzer = UnifiedPlatformAnalyzer(
            config_path=args.config if Path(args.config).exists() else None,
            output_dir=args.output_dir
        )
        
        # Run pipeline
        success = analyzer.run_pipeline(
            steps=args.steps,
            resume_from=args.resume_from
        )
        
        if success:
            logger.info("🎉 Analysis completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Analysis failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
