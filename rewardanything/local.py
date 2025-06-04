"""Local inference implementation for RewardAnything models."""

import torch
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .models import RewardResult
from .processing import prepare_chat_messages, parse_rewardanything_output
import logging

logger = logging.getLogger(__name__)


class RewardModel:
    """Local RewardAnything model for principle-following evaluation."""
    
    def __init__(
        self,
        model,
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def judge(
        self,
        principle: str,
        prompt: str,
        responses: Dict[str, str],
        mask_responses: bool = True,
        **generation_kwargs
    ) -> RewardResult:
        """
        Evaluate responses based on a natural language principle.
        
        Args:
            principle: Natural language principle for evaluation
            prompt: The input prompt that responses are answering
            responses: Dict mapping model names to their responses
            mask_responses: Whether to mask model names during evaluation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            RewardResult containing scores, ranking, and reasoning
        """
        # Prepare chat messages using unified processing
        messages, masked2real = prepare_chat_messages(
            principle, prompt, responses, mask_responses
        )
        
        # Format for the model
        formatted_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate response
        generation_config = self.generation_config
        if generation_kwargs:
            generation_config = GenerationConfig(**{
                **self.generation_config.to_dict(),
                **generation_kwargs
            })
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse output using unified processing
        return parse_rewardanything_output(output_text, masked2real)
    
    def judge_batch(
        self,
        requests: List[Dict[str, Any]],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[RewardResult]:
        """
        Evaluate multiple requests in batches.
        
        Args:
            requests: List of dicts with 'principle', 'prompt', 'responses' keys
            batch_size: Batch size for processing
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of RewardResult objects
        """
        results = []
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            for request in batch:
                result = self.judge(
                    principle=request["principle"],
                    prompt=request["prompt"],
                    responses=request["responses"],
                    mask_responses=request.get("mask_responses", True),
                    **generation_kwargs
                )
                results.append(result)
        return results


def from_pretrained(
    model_name_or_path: str,
    device: Optional[str] = None,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
    trust_remote_code: bool = False,
    generation_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RewardModel:
    """
    Load a RewardAnything model for local inference.
    
    Args:
        model_name_or_path: Path to model or HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', 'auto')
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
        generation_config: Generation configuration parameters
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
        
    Returns:
        RewardModel instance ready for evaluation
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **{k: v for k, v in kwargs.items() if k in ['use_fast', 'padding_side']}
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else "auto",
        trust_remote_code=trust_remote_code,
        **{k: v for k, v in kwargs.items() if k not in ['use_fast', 'padding_side']}
    )
    
    # Create generation config
    gen_config = None
    if generation_config:
        gen_config = GenerationConfig(**generation_config)
    
    return RewardModel(
        model=model,
        tokenizer=tokenizer,
        generation_config=gen_config,
        device=device
    )