"""Unified input preparation and output parsing for RewardAnything."""

import json
import random
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import kendalltau, spearmanr
from .models import RewardRequest, RewardResult

# System prompt for RewardAnything evaluation
SYSTEM_PROMPT = """
You are an evaluator judging model responses based on a given **evaluation principle**. Your primary goal is to assess how well each response for the prompt adheres to the principle, placing this above typical general preferences, though you should not endorse harmful content.

Your task:
1. Read the principle, prompt, and all responses carefully and consider how each response aligns with the principle, **briefly in a concise thinking process**
2. Score each response from 1-5:
   - 5: Perfect adherence + excellent quality
   - 4: Strong adherence with minor limitations
   - 3: Basic adherence
   - 2: Partial adherence with key omissions
   - 1: Poor adherence or contradicts principle
3. Sort responses from best to worst (distinguish between same scores)

Use the scoring scale accurately based on merit - don't compress scores if responses show significant quality differences. If responses vary substantially in quality, utilize the full range (1-5) to reflect these differences.

Output ONLY this JSON format:
{
  "scores": {"model-1": 2, "model-2": 4, ...},
  "best-to-worst": ["model-2", "model-1", ...]
}
"""


def make_masked_responses(responses: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Create masked model names to prevent bias in evaluation.
    Returns masked responses and a mapping from masked to real names.
    """
    masked_names = [f'model-{i+1}' for i in range(len(responses))]
    realname_responses = list(responses.items())
    random.seed(42)  # For reproducibility
    random.shuffle(realname_responses)
    
    masked2real = {masked_names[i]: realname_responses[i][0] for i in range(len(realname_responses))}
    responses_masked = {masked_names[i]: realname_responses[i][1] for i in range(len(realname_responses))}
    
    return responses_masked, masked2real


def prepare_chat_messages(
    principle: str,
    prompt: str,
    responses: Dict[str, str],
    mask_responses: bool = True
) -> Tuple[List[Dict[str, str]], Optional[Dict[str, str]]]:
    """
    Prepare chat messages for RewardAnything evaluation.
    
    Returns:
        Tuple of (messages, masked2real_mapping)
    """
    # Apply masking if requested
    masked2real = None
    if mask_responses:
        responses_to_use, masked2real = make_masked_responses(responses)
    else:
        responses_to_use = responses
    
    # Create user content as JSON
    user_content = json.dumps({
        "principle": principle,
        "prompt": prompt,
        "responses": responses_to_use,
    }, ensure_ascii=False)
    
    # Construct messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    return messages, masked2real


def extract_json_content(text: str) -> str:
    """Extract JSON content after the </think> tag."""
    # First remove the thinking section
    after_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Clean up any markdown code block indicators
    clean_text = re.sub(r'```(?:json)?|```', '', after_think)
    
    # Return the cleaned text that should contain just JSON
    return clean_text.strip()


def fix_json_format(json_str: str) -> str:
    """Fix common JSON formatting issues."""
    # First, replace all single quotes with double quotes
    # This is a simple approach but works for most cases
    fixed = json_str.replace("'", '"')
    
    # Add quotes to unquoted property names (if any remain)
    fixed = re.sub(r'([{,])\s*([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', fixed)
    
    # Handle trailing commas which are invalid in JSON
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    
    # Fix common typos in keys
    fixed = fixed.replace('"best-to-worse"', '"best-to-worst"')
    
    return fixed


def extract_json_by_brackets(text: str) -> Optional[str]:
    """Extract JSON by finding first { and last }."""
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]
    return None


def safe_parse_json(text: str) -> Tuple[Dict, bool]:
    """
    Safely parse JSON with multiple fallbacks.
    Returns (parsed_dict, used_bracket_extraction)
    """
    # First attempt: direct parsing
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    
    # Second attempt: fix common formatting issues
    try:
        fixed = fix_json_format(text)
        return json.loads(fixed), False
    except:
        pass
    
    # Third attempt: extract by brackets
    try:
        bracket_json = extract_json_by_brackets(text)
        if bracket_json:
            # Try to parse the bracket-extracted content
            result = json.loads(bracket_json)
            if isinstance(result, dict):
                return result, True
    except:
        pass
    
    # Final fallback
    return {}, False


def calculate_weighted_pair_penalty(pred_scores: Dict[str, int], gt_scores: Dict[str, int]) -> float:
    """
    Calculate weighted pair reversal penalty.
    Higher penalty for reversing pairs with larger score differences.
    """
    common_models = list(set(pred_scores.keys()) & set(gt_scores.keys()))
    
    if len(common_models) < 2:
        return 1.0  # No pairs to compare
        
    total_penalty = 0.0
    total_weight = 0.0
    
    # Compare all pairs
    for i in range(len(common_models)):
        for j in range(i + 1, len(common_models)):
            model_a, model_b = common_models[i], common_models[j]
            
            try:
                # Get scores
                pred_a = float(pred_scores[model_a])
                pred_b = float(pred_scores[model_b])
                gt_a = float(gt_scores[model_a])
                gt_b = float(gt_scores[model_b])
                
                # Calculate ground truth difference (weight)
                gt_diff = abs(gt_a - gt_b)
                weight = 1 + gt_diff  # Base weight of 1 plus difference
                
                # Check if pair is correctly ordered
                pred_order = pred_a - pred_b
                gt_order = gt_a - gt_b
                
                # If signs differ, the pair is reversed
                if pred_order * gt_order < 0:
                    penalty = weight
                elif pred_order == 0 and gt_order != 0:
                    # Predicted tie when there shouldn't be one
                    penalty = weight * 0.5
                else:
                    penalty = 0
                    
                total_penalty += penalty
                total_weight += weight
                
            except (ValueError, TypeError):
                continue
                
    if total_weight == 0:
        return 0.0
        
    # Convert to reward (higher is better)
    return max(0, 1 - total_penalty / total_weight)


def format_reward(predict: str, ground_truth: str) -> float:
    """
    Enhanced format reward with more granular scoring:
    1. Thinking tags presence and quality
    2. JSON structure validity
    3. Required keys completeness
    4. Model coverage
    5. Consistency between scores and best-to-worst
    """
    score = 0.0
    
    # Check for thinking section (20%)
    has_think_open = '<think>' in predict
    has_think_close = '</think>' in predict
    if has_think_open and has_think_close:
        score += 0.15
        # Bonus for substantial thinking (5%)
        think_match = re.search(r'<think>(.*?)</think>', predict, re.DOTALL)
        if think_match and len(think_match.group(1).strip()) > 50:
            score += 0.05
    elif has_think_open or has_think_close:
        # Partial credit for attempting
        score += 0.05
    
    # Extract and parse JSON
    json_content = extract_json_content(predict)
    pred_json, used_bracket_extraction = safe_parse_json(json_content)
    gt_json, _ = safe_parse_json(ground_truth)
    
    # Valid JSON structure (20%)
    if pred_json:
        # Apply penalty if bracket extraction was used
        json_score = 0.20 if not used_bracket_extraction else 0.15
        score += json_score
        
        # Required keys (20%)
        has_scores = "scores" in pred_json
        # Accept both "best-to-worst" and "best-to-worse"
        has_ranking = "best-to-worst" in pred_json or "best-to-worse" in pred_json
        
        if has_scores:
            score += 0.10
        if has_ranking:
            score += 0.10
            
        # Model coverage (20%)
        if has_scores and "scores" in gt_json:
            gt_models = set(gt_json["scores"].keys())
            pred_models = set(pred_json["scores"].keys()) if isinstance(pred_json["scores"], dict) else set()
            
            if gt_models and pred_models:
                coverage = len(gt_models & pred_models) / max(1, len(gt_models))
                score += 0.20 * coverage
                
        # Consistency check (20%)
        if has_scores and has_ranking:
            consistency_score = check_ranking_consistency(pred_json)
            score += 0.20 * consistency_score
    
    return min(1.0, score)


def parse_rewardanything_output(
    output_text: str,
    masked2real: Optional[Dict[str, str]] = None
) -> RewardResult:
    """
    Parse RewardAnything model output into structured result.
    
    Args:
        output_text: Raw output from the model
        masked2real: Optional mapping from masked names to real names
        
    Returns:
        RewardResult with scores, ranking, and reasoning
    """
    # Extract thinking section
    think_start = output_text.find("<think>")
    think_end = output_text.find("</think>")
    
    if think_start != -1 and think_end != -1:
        reasoning = output_text[think_start + 7:think_end].strip()
    else:
        reasoning = ""
    
    # Extract and parse JSON
    json_content = extract_json_content(output_text)
    results, _ = safe_parse_json(json_content)
    
    # Ensure the results have the expected structure
    if not results:
        results = {"scores": {}, "best-to-worst": []}
    elif "scores" not in results:
        results["scores"] = {}
    elif "best-to-worst" not in results:
        # Handle both key variants
        if "best-to-worse" in results:
            results["best-to-worst"] = results["best-to-worse"]
        else:
            results["best-to-worst"] = []
    
    scores = results.get("scores", {})
    ranking = results.get("best-to-worst", [])
    
    # Convert masked names back to real names if needed
    if masked2real:
        # Convert scores
        real_scores = {}
        for masked_name, score in scores.items():
            if masked_name in masked2real:
                real_scores[masked2real[masked_name]] = score
            else:
                real_scores[masked_name] = score  # Keep unmapped names as-is
        
        # Convert ranking
        real_ranking = []
        for masked_name in ranking:
            if masked_name in masked2real:
                real_ranking.append(masked2real[masked_name])
            else:
                real_ranking.append(masked_name)  # Keep unmapped names as-is
                
        scores = real_scores
        ranking = real_ranking
    
    return RewardResult(
        reasoning=reasoning,
        scores=scores,
        ranking=ranking,
        raw_output=output_text
    )


def analyze_prediction_quality(predict: str, ground_truth: str) -> Dict:
    """
    Detailed analysis of a single prediction for debugging.
    """
    try:
        pred_json, used_brackets = safe_parse_json(extract_json_content(predict))
        gt_json, _ = safe_parse_json(ground_truth)
        
        analysis = {
            "has_thinking": '<think>' in predict and '</think>' in predict,
            "valid_json": bool(pred_json),
            "used_bracket_extraction": used_brackets,
            "has_required_keys": bool(pred_json.get("scores")) and 
                               (bool(pred_json.get("best-to-worst")) or bool(pred_json.get("best-to-worse"))),
            "model_coverage": 0.0,
            "score_differences": {},
            "reversed_pairs": []
        }
        
        if pred_json and gt_json:
            pred_scores = pred_json.get("scores", {})
            gt_scores = gt_json.get("scores", {})
            
            common = set(pred_scores.keys()) & set(gt_scores.keys())
            if gt_scores:
                analysis["model_coverage"] = len(common) / len(gt_scores)
                
            # Analyze score differences
            for model in common:
                try:
                    analysis["score_differences"][model] = {
                        "predicted": float(pred_scores[model]),
                        "ground_truth": float(gt_scores[model]),
                        "difference": abs(float(pred_scores[model]) - float(gt_scores[model]))
                    }
                except:
                    pass
                    
            # Find reversed pairs
            models = list(common)
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    try:
                        pred_diff = float(pred_scores[models[i]]) - float(pred_scores[models[j]])
                        gt_diff = float(gt_scores[models[i]]) - float(gt_scores[models[j]])
                        
                        if pred_diff * gt_diff < 0:
                            analysis["reversed_pairs"].append({
                                "pair": (models[i], models[j]),
                                "weight": abs(gt_diff)
                            })
                    except:
                        pass
                        
        return analysis
        
    except:
        return {"error": "Analysis failed"} 