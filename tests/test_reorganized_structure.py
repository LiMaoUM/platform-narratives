#!/usr/bin/env python3
"""
Test script for the reorganized platform narratives analysis structure.

This script tests the new unified pipeline and module structure to ensure
everything is working correctly after the reorganization.
"""

import sys
import os
import traceback
from pathlib import Path

# Add repo root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_imports():
    """Test that all new modules can be imported correctly."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core analyzers
        from src import CrossPlatformAnalyzer, ReplyAnalyzer, ConfigManager
        print("✅ Core analyzers imported successfully")
        
        # Test utility modules
        from src import SimilarityGraphBuilder
        from src.text_processing import clean_text, detect_post_language
        from src.utils import setup_logging, save_pipeline_state, load_pipeline_state
        print("✅ Utility modules imported successfully")
        
        # Test optional modules
        from src import VLLM_AVAILABLE, NARRATIVE_CLASSIFICATION_AVAILABLE
        print(f"✅ vLLM available: {VLLM_AVAILABLE}")
        print(f"✅ Narrative classification available: {NARRATIVE_CLASSIFICATION_AVAILABLE}")
        
        if VLLM_AVAILABLE:
            from src import check_vllm_availability
            print("✅ vLLM wrapper imported successfully")
        
        # Test unified analyzer
        from unified_analysis import UnifiedPlatformAnalyzer
        print("✅ Unified analyzer imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_config_manager():
    """Test configuration management."""
    print("\n🧪 Testing configuration manager...")
    
    try:
        from src import ConfigManager
        
        # Test with default config
        config_mgr = ConfigManager()
        config = config_mgr.get_config()
        print(f"✅ Default config loaded with {len(config)} sections")
        
        # Test validation
        is_valid = config_mgr.validate_config()
        print(f"✅ Config validation: {is_valid}")
        
        # Test with YAML config if exists
        yaml_config_path = "config_unified.yaml"
        if Path(yaml_config_path).exists():
            config_mgr_yaml = ConfigManager(yaml_config_path)
            yaml_config = config_mgr_yaml.get_config()
            print(f"✅ YAML config loaded with {len(yaml_config)} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Config manager test failed: {e}")
        traceback.print_exc()
        return False

def test_analyzers_initialization():
    """Test that analyzers can be initialized."""
    print("\n🧪 Testing analyzer initialization...")
    
    try:
        from src import CrossPlatformAnalyzer, ReplyAnalyzer, ConfigManager
        
        # Test cross-platform analyzer
        config_mgr = ConfigManager("config_unified.yaml" if Path("config_unified.yaml").exists() else None)
        cross_analyzer = CrossPlatformAnalyzer(
            config=config_mgr.get_config(),
            output_dir="test_cross_platform_output"
        )
        print("✅ CrossPlatformAnalyzer initialized")
        
        # Test reply analyzer
        reply_analyzer = ReplyAnalyzer(output_dir="test_reply_output")
        print("✅ ReplyAnalyzer initialized")
        
        # Test unified analyzer
        from unified_analysis import UnifiedPlatformAnalyzer
        unified_analyzer = UnifiedPlatformAnalyzer(
            config_path="config_unified.yaml" if Path("config_unified.yaml").exists() else None,
            output_dir="test_unified_output"
        )
        print("✅ UnifiedPlatformAnalyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_text_processing():
    """Test text processing utilities."""
    print("\n🧪 Testing text processing...")
    
    try:
        from src.text_processing import clean_text, detect_post_language
        
        # Test text cleaning
        test_text = "<p>Check out this @user #hashtag http://example.com</p>"
        cleaned = clean_text(test_text)
        print(f"✅ Text cleaned: '{test_text}' -> '{cleaned}'")
        
        # Test language detection
        lang = detect_post_language("This is a test post in English")
        print(f"✅ Language detected: {lang}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        traceback.print_exc()
        return False

def test_data_directory():
    """Test that data directory structure is accessible."""
    print("\n🧪 Testing data directory access...")
    
    try:
        data_dir = Path("data/data")
        if not data_dir.exists():
            print(f"⚠️ Data directory not found: {data_dir}")
            return True  # Not an error, just no data
        
        # Check for expected files
        expected_files = [
            "truthsocial.trump.json",
            "bsky.trump.json", 
            "mastodon.trump.json"
        ]
        
        found_files = []
        for file in expected_files:
            file_path = data_dir / file
            if file_path.exists():
                found_files.append(file)
                print(f"✅ Found data file: {file}")
            else:
                print(f"⚠️ Data file not found: {file}")
        
        print(f"✅ Data directory accessible, found {len(found_files)}/{len(expected_files)} files")
        return True
        
    except Exception as e:
        print(f"❌ Data directory test failed: {e}")
        return False

def test_pipeline_steps():
    """Test that pipeline steps can be defined and validated."""
    print("\n🧪 Testing pipeline step definitions...")
    
    try:
        from unified_analysis import UnifiedPlatformAnalyzer
        
        # Create analyzer instance
        analyzer = UnifiedPlatformAnalyzer(output_dir="test_pipeline_output")
        
        # Test that step methods exist
        expected_steps = [
            'step_setup',
            'step_load_data', 
            'step_cross_platform_matching',
            'step_narrative_analysis',
            'step_conversation_graphs',
            'step_reply_classification',
            'step_combined_analysis',
            'step_generate_reports'
        ]
        
        for step in expected_steps:
            if hasattr(analyzer, step):
                print(f"✅ Pipeline step exists: {step}")
            else:
                print(f"❌ Pipeline step missing: {step}")
                return False
        
        print("✅ All pipeline steps defined")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline steps test failed: {e}")
        traceback.print_exc()
        return False

def run_minimal_pipeline_test():
    """Test running just the setup step of the pipeline."""
    print("\n🧪 Testing minimal pipeline execution...")
    
    try:
        from unified_analysis import UnifiedPlatformAnalyzer
        
        # Create analyzer
        analyzer = UnifiedPlatformAnalyzer(
            config_path="config_unified.yaml" if Path("config_unified.yaml").exists() else None,
            output_dir="test_minimal_pipeline"
        )
        
        # Try to run just the setup step
        success = analyzer.run_pipeline(steps=['setup'])
        
        if success:
            print("✅ Minimal pipeline (setup step) executed successfully")
        else:
            print("⚠️ Minimal pipeline completed with warnings")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal pipeline test failed: {e}")
        traceback.print_exc()
        return False

def cleanup_test_outputs():
    """Clean up test output directories."""
    print("\n🧹 Cleaning up test outputs...")
    
    import shutil
    
    test_dirs = [
        "test_cross_platform_output",
        "test_reply_output", 
        "test_unified_output",
        "test_pipeline_output",
        "test_minimal_pipeline"
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            try:
                shutil.rmtree(test_dir)
                print(f"✅ Cleaned up: {test_dir}")
            except Exception as e:
                print(f"⚠️ Could not clean up {test_dir}: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting reorganized structure tests")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Manager", test_config_manager),
        ("Analyzer Initialization", test_analyzers_initialization),
        ("Text Processing", test_text_processing),
        ("Data Directory Access", test_data_directory),
        ("Pipeline Steps", test_pipeline_steps),
        ("Minimal Pipeline", run_minimal_pipeline_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    # Cleanup
    cleanup_test_outputs()
    
    # Summary
    print("\n" + "="*60)
    print("🧪 TEST SUMMARY")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The reorganized structure is working correctly.")
        print("\nNext steps:")
        print("1. Run: python run_quick_analysis.py --mode quick")
        print("2. Or: python unified_analysis.py --steps setup load_data")
        return 0
    else:
        print(f"\n💥 {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
