#!/usr/bin/env python3
"""
Simple runner script for cross-platform narrative analysis.
This provides a more user-friendly interface to the main analysis script.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from narrative_analysis_workflow import NarrativeAnalysisWorkflow, create_sample_platform_data
from config import ANALYSIS_CONFIGS, PLATFORM_CONFIGS, NARRATIVE_FRAMES

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'results', 'figures', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

def run_sample_analysis(config_name: str = "quick_test"):
    """Run analysis with sample data."""
    print("Running sample analysis...")
    
    # Get configuration
    config = ANALYSIS_CONFIGS.get(config_name, ANALYSIS_CONFIGS["quick_test"])
    
    # Initialize workflow
    workflow = NarrativeAnalysisWorkflow(
        similarity_threshold=config["similarity_threshold"],
        model_name=config["model"],
        output_dir="results"
    )
    
    # Create sample data
    sample_data = create_sample_platform_data()
    
    # Run analysis
    results = workflow.run_complete_analysis(sample_data, f"sample_{config_name}")
    
    print(f"Sample analysis completed! Results saved to results/sample_{config_name}_*")
    return results

def run_file_analysis(file_path: str, config_name: str = "full_analysis", experiment_name: Optional[str] = None):
    """Run analysis with data from file."""
    print(f"Running analysis with data from {file_path}...")
    
    # Get configuration
    config = ANALYSIS_CONFIGS.get(config_name, ANALYSIS_CONFIGS["full_analysis"])
    
    # Initialize workflow
    workflow = NarrativeAnalysisWorkflow(
        similarity_threshold=config["similarity_threshold"],
        model_name=config["model"],
        output_dir="results"
    )
    
    # Load data
    try:
        platform_data = workflow.load_platform_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Set experiment name
    if experiment_name is None:
        experiment_name = f"analysis_{Path(file_path).stem}_{config_name}"
    
    # Run analysis
    results = workflow.run_complete_analysis(platform_data, experiment_name)
    
    print(f"File analysis completed! Results saved to results/{experiment_name}_*")
    return results

def compare_experiments(experiment_names: List[str]):
    """Compare multiple experiments."""
    print(f"Comparing experiments: {', '.join(experiment_names)}")
    
    workflow = NarrativeAnalysisWorkflow(output_dir="results")
    comparison_df = workflow.compare_experiments(experiment_names)
    
    print("\nExperiment Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def list_available_configs():
    """List available analysis configurations."""
    print("Available analysis configurations:")
    for name, config in ANALYSIS_CONFIGS.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

def show_narrative_frames():
    """Display the narrative frame taxonomy."""
    print("Narrative Frame Taxonomy:")
    for frame_name, frame_info in NARRATIVE_FRAMES.items():
        print(f"\n{frame_name.upper()}:")
        print(f"  Definition: {frame_info['definition']}")
        print(f"  Keywords: {', '.join(frame_info['keywords'])}")
        if frame_info.get('examples'):
            print(f"  Examples:")
            for example in frame_info['examples']:
                print(f"    - {example}")

def create_sample_data_file(output_path: str = "data/sample_data.json"):
    """Create a sample data file for testing."""
    sample_data = create_sample_platform_data()
    
    # Ensure data directory exists
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Save sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample data file created at {output_path}")
    return output_path

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Narrative Divergence Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py sample                          # Run with sample data
  python run_analysis.py file data/posts.json           # Analyze data from file
  python run_analysis.py compare exp1 exp2 exp3         # Compare experiments
  python run_analysis.py configs                        # List available configs
  python run_analysis.py frames                         # Show narrative frames
  python run_analysis.py create-sample                  # Create sample data file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sample analysis command
    sample_parser = subparsers.add_parser('sample', help='Run analysis with sample data')
    sample_parser.add_argument('--config', default='quick_test', 
                              choices=list(ANALYSIS_CONFIGS.keys()),
                              help='Analysis configuration to use')
    
    # File analysis command  
    file_parser = subparsers.add_parser('file', help='Run analysis with data from file')
    file_parser.add_argument('path', help='Path to data file (JSON, CSV, or pickle)')
    file_parser.add_argument('--config', default='full_analysis',
                            choices=list(ANALYSIS_CONFIGS.keys()),
                            help='Analysis configuration to use')
    file_parser.add_argument('--name', help='Custom experiment name')
    
    # Compare experiments command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple experiments')
    compare_parser.add_argument('experiments', nargs='+', 
                               help='Names of experiments to compare')
    
    # List configs command
    subparsers.add_parser('configs', help='List available analysis configurations')
    
    # Show frames command
    subparsers.add_parser('frames', help='Display narrative frame taxonomy')
    
    # Create sample data command
    create_parser = subparsers.add_parser('create-sample', help='Create sample data file')
    create_parser.add_argument('--output', default='data/sample_data.json',
                              help='Output path for sample data file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Execute commands
    if args.command == 'sample':
        run_sample_analysis(args.config)
    
    elif args.command == 'file':
        run_file_analysis(args.path, args.config, args.name)
    
    elif args.command == 'compare':
        compare_experiments(args.experiments)
    
    elif args.command == 'configs':
        list_available_configs()
    
    elif args.command == 'frames':
        show_narrative_frames()
    
    elif args.command == 'create-sample':
        create_sample_data_file(args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
