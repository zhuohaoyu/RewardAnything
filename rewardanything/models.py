"""Data models and result classes for RewardAnything."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class RewardResult:
    """Result from RewardAnything evaluation."""
    reasoning: str
    scores: Dict[str, float]  # model_name -> score (1-5)
    ranking: List[str]        # ordered list from best to worst
    raw_output: Optional[str] = None
    
    def __str__(self) -> str:
        return f"RewardResult(scores={self.scores}, ranking={self.ranking})"
    
    def __repr__(self) -> str:
        return self.__str__()


class RewardRequest(BaseModel):
    """Request format for RewardAnything evaluation."""
    principle: str
    prompt: str
    responses: Dict[str, str]
    mask_responses: bool = True


class RewardResponse(BaseModel):
    """Response format from RewardAnything server."""
    thoughts: str
    results: Dict[str, Any]