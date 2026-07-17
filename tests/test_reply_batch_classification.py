#!/usr/bin/env python3
"""
Test script for reply classification batch processing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'legacy'))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from reply_analysis_complete import ReplyAnalyzerWithVLLM

def test_reply_batch_classification():
    """Test the batch classification functionality."""
    
    print("Testing reply classification batch processing...")
    
    # Initialize analyzer
    analyzer = ReplyAnalyzerWithVLLM()
    
    # Test data - (root_post, reply_post) pairs
    test_pairs = [
        ("Trump's policies will make America great again!", "I completely agree, his economic plan is brilliant."),
        ("Climate change is a serious threat to our planet.", "That's just a hoax by the liberal media."),
        ("The new movie was amazing!", "I haven't seen it yet, but I heard the soundtrack is good."),
    ]
    
    print(f"Testing with {len(test_pairs)} reply pairs...")
    
    try:
        # Test the batch classification method directly
        results = analyzer._batch_classify_replies(test_pairs)
        
        print(f"Results count: {len(results)}")
        for i, (pair, result) in enumerate(zip(test_pairs, results)):
            root, reply = pair
            print(f"\nPair {i+1}:")
            print(f"  Root: {root[:50]}...")
            print(f"  Reply: {reply[:50]}...")
            print(f"  Generated Text: {result}...")
            print(f"  Classification: {result['classification']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Reasoning: {result['reasoning'][:100]}...")
        
        print("\n✅ Batch classification test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_reply_batch_classification()
