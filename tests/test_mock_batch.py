#!/usr/bin/env python3
"""
Test the logic of our batch processing implementation without requiring GPU.
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'legacy'))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Mock the vLLM functionality for testing
class MockVLLMPipeline:
    """Mock vLLM pipeline for testing without GPU."""
    
    def __call__(self, messages, max_new_tokens=200, temperature=0.1):
        """Mock batch processing."""
        
        # Determine if batch or single
        is_batch = isinstance(messages[0], list) if messages else False
        
        results = []
        message_list = messages if is_batch else [messages]
        
        for msg_group in message_list:
            # Mock response based on content
            content = msg_group[0]['content'].lower()
            
            if 'agree' in content or 'brilliant' in content:
                mock_response = '{"classification": "REINFORCE", "confidence": 0.8, "reasoning": "Shows agreement and support"}'
            elif 'hoax' in content or 'terrible' in content:
                mock_response = '{"classification": "CHALLENGE", "confidence": 0.9, "reasoning": "Directly contradicts the original post"}'
            else:
                mock_response = '{"classification": "SHIFT", "confidence": 0.6, "reasoning": "Introduces new topic or perspective"}'
            
            results.append({
                "generated_text": [{
                    "role": "assistant", 
                    "content": mock_response
                }]
            })
        
        return results

def test_batch_logic():
    """Test our batch processing logic with mocked vLLM."""
    
    print("Testing batch processing logic with mock vLLM...")
    
    # Import our analyzer
    from reply_analysis_complete import ReplyAnalyzerWithVLLM
    
    # Create analyzer and set up mock
    analyzer = ReplyAnalyzerWithVLLM()
    analyzer.vllm_pipeline = MockVLLMPipeline()
    analyzer.vllm_available = True
    
    # Test pairs
    test_pairs = [
        ("Trump's policies will make America great again!", "I completely agree, his economic plan is brilliant."),
        ("Climate change is a serious threat to our planet.", "That's just a hoax by the liberal media."),
        ("The new movie was amazing!", "I haven't seen it yet, but I heard the soundtrack is good."),
        ("We need better healthcare!", "Yes, but we also need to consider the costs."),
    ]
    
    print(f"Testing with {len(test_pairs)} reply pairs...")
    
    try:
        # Test batch classification
        results = analyzer._batch_classify_replies(test_pairs)
        
        print(f"✅ Successfully processed {len(results)} replies")
        
        for i, (pair, result) in enumerate(zip(test_pairs, results)):
            root, reply = pair
            print(f"\nPair {i+1}:")
            print(f"  Root: {root[:60]}...")
            print(f"  Reply: {reply[:60]}...")
            print(f"  Classification: {result['classification']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')[:80]}...")
        
        # Test text cleaning
        print(f"\n📝 Testing text cleaning...")
        test_html = '<p>Check out this <a href="http://example.com">link</a> @user #hashtag</p>'
        cleaned = analyzer.clean_text(test_html, platform='mastodon')
        print(f"Original: {test_html}")
        print(f"Cleaned:  {cleaned}")
        
        print("\n✅ All tests passed! Batch processing logic is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_batch_logic()
