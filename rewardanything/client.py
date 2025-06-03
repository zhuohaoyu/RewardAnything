"""Remote client implementation for RewardAnything."""

import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
from .models import RewardResult, RewardRequest, RewardResponse
import logging

logger = logging.getLogger(__name__)


class Client:
    """Remote client for RewardAnything API."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize RewardAnything client.
        
        Args:
            base_url: Base URL for the RewardAnything API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            headers: Additional headers to include in requests
            **kwargs: Additional client configuration
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "RewardAnything-Python-Client/1.0.0"
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
            
        if headers:
            self.headers.update(headers)
            
        # Store additional config
        self.config = kwargs
        
    def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retries."""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=data,
                    headers=self.headers,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
                    
        raise last_exception
    
    def judge(
        self,
        principle: str,
        prompt: str,
        responses: Dict[str, str],
        mask_responses: bool = True,
        timeout: Optional[float] = None,
        **kwargs
    ) -> RewardResult:
        """
        Evaluate responses based on a natural language principle.
        
        Args:
            principle: Natural language principle for evaluation
            prompt: The input prompt that responses are answering
            responses: Dict mapping model names to their responses
            mask_responses: Whether to mask model names during evaluation
            timeout: Request timeout override
            **kwargs: Additional request parameters
            
        Returns:
            RewardResult containing scores, ranking, and reasoning
        """
        request_data = {
            "principle": principle,
            "prompt": prompt,
            "responses": responses,
            "mask_responses": mask_responses
        }
        
        # Add any additional config
        request_data.update(kwargs)
        
        try:
            response_data = self._make_request(
                "/api/rewardanything",
                request_data,
                timeout
            )
            
            # Parse response
            thoughts = response_data.get("thoughts", "")
            results = response_data.get("results", {})
            
            return RewardResult(
                reasoning=thoughts,
                scores=results.get("scores", {}),
                ranking=results.get("best-to-worst", []),
                raw_output=json.dumps(response_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate with principle '{principle}': {e}")
            raise
    
    def judge_batch(
        self,
        requests: List[Dict[str, Any]],
        timeout: Optional[float] = None,
        **kwargs
    ) -> List[RewardResult]:
        """
        Evaluate multiple requests in a batch.
        
        Args:
            requests: List of dicts with 'principle', 'prompt', 'responses' keys
            timeout: Request timeout override
            **kwargs: Additional request parameters
            
        Returns:
            List of RewardResult objects
        """
        # Convert to RewardRequest format
        batch_requests = []
        for req in requests:
            batch_requests.append({
                "principle": req["principle"],
                "prompt": req["prompt"],
                "responses": req["responses"],
                "mask_responses": req.get("mask_responses", True)
            })
        
        try:
            response_data = self._make_request(
                "/api/rewardanything_batch",
                batch_requests,
                timeout or (self.timeout * len(requests))  # Scale timeout with batch size
            )
            
            # Parse batch response
            results = []
            for item in response_data:
                results.append(RewardResult(
                    reasoning=item.get("thoughts", ""),
                    scores=item.get("results", {}).get("scores", {}),
                    ranking=item.get("results", {}).get("best-to-worst", []),
                    raw_output=json.dumps(item)
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate batch of {len(requests)} requests: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False