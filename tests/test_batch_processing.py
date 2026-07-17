#!/usr/bin/env python3
"""
Test script for batch processing functionality in vLLM wrapper.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from vllm_wrapper import check_vllm_availability, create_vllm_pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_processing():
    """Test the batch processing functionality of the vLLM wrapper."""
    
    print("Testing vLLM batch processing...")
    
    # Check if vLLM is available
    if not check_vllm_availability():
        print("❌ vLLM not available. Please install with: pip install vllm")
        return False
    
    try:
        # Create a smaller model for testing to avoid memory issues
        print("Creating vLLM pipeline for testing...")
        pipeline = create_vllm_pipeline(
            model_name="microsoft/DialoGPT-medium",  # Use a smaller model for testing
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=1024
        )
        
        # Test single request (existing functionality)
        print("\n1. Testing single request...")
        single_messages = [{"role": "user", "content": "Say hello"}]
        single_response = pipeline(single_messages, max_new_tokens=50)
        print(f"Single response: {single_response[0]['generated_text'][0]['content'][:100]}...")
        
        # Test batch processing (new functionality)
        print("\n2. Testing batch processing...")
        batch_messages = [
            [{"role": "user", "content": "Classify this as positive or negative: I love this!"}],
            [{"role": "user", "content": "Classify this as positive or negative: This is terrible."}],
            [{"role": "user", "content": "Classify this as positive or negative: It's okay, I guess."}]
        ]
        
        batch_responses = pipeline(batch_messages, max_new_tokens=50)
        
        print(f"Batch response count: {len(batch_responses)}")
        for i, response in enumerate(batch_responses):
            content = response['generated_text'][0]['content']
            print(f"Response {i+1}: {content[:100]}...")
        
        print("✅ Batch processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_batch_processing()
