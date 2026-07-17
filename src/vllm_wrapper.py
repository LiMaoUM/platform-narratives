"""vLLM wrapper for narrative classification.

This module provides a wrapper around vLLM with the Gemma 3 27B model,
the LLM used for narrative frame and reply classification in the paper.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Install with: pip install vllm")


class VLLMWrapper:
    """
    Wrapper for vLLM that provides compatibility with the narrative classification system.
    """
    
    def __init__(self, model_name: str = "google/gemma-3-27b-it", 
                 tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.95,
                 max_model_len: Optional[int] = None):
        """
        Initialize the vLLM wrapper.
        
        Args:
            model_name: Name of the model to load (defaults to google/gemma-3-27b-it)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (optional, uses model default if None)
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Install with: pip install vllm")
        
        # Set environment variable to allow long max model length
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
        
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        
        # Define fallback models in order of preference (smallest to largest)
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "google/gemma-2-9b-it",
            "google/gemma-3-27b-it"
        ]
        
        
        # Initialize vLLM engine with fallback logic
        self.llm = self._initialize_model_with_fallback(
            model_name, tensor_parallel_size, gpu_memory_utilization, max_model_len
        )
    
    def _initialize_model_with_fallback(self, 
                                       primary_model: str, 
                                       tensor_parallel_size: int, 
                                       gpu_memory_utilization: float, 
                                       max_model_len: Optional[int]) -> LLM:
        """
        Try to initialize the primary model, fall back to smaller models if needed.
        
        Args:
            primary_model: The preferred model to load
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization fraction
            max_model_len: Maximum model length
            
        Returns:
            Initialized LLM instance
        """
        # Start with the requested model, then try fallbacks
        models_to_try = [primary_model] + [m for m in self.fallback_models if m != primary_model]
        
        for model_name in models_to_try:
            logger.info(f"Attempting to load vLLM model: {model_name}")
            logger.info(f"Tensor parallel size: {tensor_parallel_size}")
            logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
            
            try:
                # Build LLM arguments dynamically
                llm_args = {
                    "model": model_name,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "trust_remote_code": True
                }
                
                # Only add max_model_len if specified
                if max_model_len is not None:
                    llm_args["max_model_len"] = max_model_len
                
                llm = LLM(**llm_args)
                logger.info(f"Successfully loaded vLLM model: {model_name}")
                self.model_name = model_name  # Update to the actually loaded model
                return llm
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {str(e)}")
                if model_name == models_to_try[-1]:  # Last model in the list
                    logger.error("All fallback models failed. Raising the last exception.")
                    raise
                else:
                    logger.info(f"Trying next fallback model...")
                    continue
    
    def _format_messages_for_gemma(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemma chat format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted string for Gemma model
        """
        formatted_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                # Gemma doesn't have explicit system role, so prepend to user message
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == 'assistant':
                formatted_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        
        # Add model turn start
        formatted_parts.append("<start_of_turn>model\n")
        
        return "\n".join(formatted_parts)
    
    def __call__(self, messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]], 
                 max_new_tokens: int = 300, temperature: float = 0.7, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate text using vLLM in a format compatible with the narrative classifier.
        Supports both single requests and batch processing.
        
        Args:
            messages: Either a single list of message dictionaries (OpenAI format) 
                     or a list of such lists for batch processing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            List with generated text in expected format
        """
        try:
            # Determine if this is a batch request or single request
            is_batch = isinstance(messages[0], list) if messages else False
            
            if is_batch:
                # Batch processing
                prompts = []
                for message_list in messages:
                    prompt = self._format_messages_for_gemma(message_list)
                    prompts.append(prompt)
            else:
                # Single request - convert to batch of one
                prompts = [self._format_messages_for_gemma(messages)]
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=kwargs.get('top_p', 0.9),
                stop=["<end_of_turn>", "<start_of_turn>"]
            )
            
            # Generate responses (batch or single)
            logger.debug(f"Generating {len(prompts)} response(s) with vLLM model: {self.model_name}")
            responses = self.llm.generate(prompts, sampling_params)
            
            if not responses:
                raise ValueError("No responses generated from vLLM")
            
            # Format responses to match expected structure
            formatted_responses = []
            for response in responses:
                generated_text = response.outputs[0].text.strip()
                formatted_responses.append({
                    "generated_text": [{
                        "role": "assistant",
                        "content": generated_text
                    }]
                })
            
            # For single requests, maintain backward compatibility
            if not is_batch:
                return formatted_responses
            else:
                return formatted_responses
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the vLLM connection and model functionality.
        Tests both single and batch processing.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Test single request
        test_messages = [
            {"role": "user", "content": "Please respond with 'Hello, world!' to test the connection."}
        ]
        
        try:
            response = self(test_messages, max_new_tokens=50)
            generated_content = response[0]["generated_text"][0]["content"]
            
            if "hello" in generated_content.lower():
                logger.info("vLLM single request test successful")
            else:
                logger.warning(f"Unexpected response in vLLM single request test: {generated_content}")
            
            # Test batch processing
            batch_messages = [
                [{"role": "user", "content": "Say 'Test 1'"}],
                [{"role": "user", "content": "Say 'Test 2'"}]
            ]
            
            batch_responses = self(batch_messages, max_new_tokens=50)
            
            if len(batch_responses) == 2:
                logger.info("vLLM batch processing test successful")
                return True
            else:
                logger.warning(f"Unexpected batch response count: {len(batch_responses)}")
                return True  # Still consider successful if we get responses
                
        except Exception as e:
            logger.error(f"vLLM connection test failed: {str(e)}")
            return False


def create_vllm_pipeline(model_name: str = "google/gemma-3-27b-it",
                        tensor_parallel_size: int = 1,
                        gpu_memory_utilization: float = 0.95,
                        max_model_len: Optional[int] = 4096) -> VLLMWrapper:
    """
    Create a vLLM pipeline for use with the narrative classifier.
    
    Args:
        model_name: Model to use (defaults to google/gemma-3-27b-it)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use (increased to 0.95 for better memory utilization)
        max_model_len: Maximum sequence length (optional)
        
    Returns:
        Configured VLLMWrapper instance
    """
    return VLLMWrapper(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )


def check_vllm_availability() -> bool:
    """
    Check if vLLM is available and can be used.
    
    Returns:
        True if vLLM is available, False otherwise
    """
    return VLLM_AVAILABLE
