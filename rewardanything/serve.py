#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Updated imports to use unified processing
from .utils import OpenAIClient, AsyncRateLimitThreadPool
from .models import RewardRequest, RewardResponse
from .processing import prepare_chat_messages, parse_rewardanything_output

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RewardAnything API")

# Store for batch tasks
batch_tasks = {}


class BatchRequest(BaseModel):
    requests: List[Dict]
    output_path: Optional[str] = None
    api_model: Optional[str] = None
    api_key: Optional[Union[str, List[str]]] = None
    api_base: Optional[Union[str, List[str]]] = None
    api_proxy: Optional[Union[str, List[str]]] = None
    api_timeout: Optional[float] = None
    api_max_retries: Optional[int] = None
    generation_config: Optional[Dict] = None
    num_workers: Optional[int] = None
    request_limit: Optional[int] = None
    request_limit_period: Optional[int] = None
    max_error_count: Optional[int] = None
    trial_run: Optional[bool] = None
    dump_individual_rsp: Optional[bool] = None


class BatchRequestResponse(BaseModel):
    batch_request_id: str
    requests: int
    start_time: str


class BatchTask:
    def __init__(self, request: BatchRequest, config: Dict, base_output_path: str):
        self.id = str(uuid.uuid4())
        self.request = request
        self.config = config
        self.base_output_path = base_output_path
        self.start_time = datetime.now().isoformat()
        self.complete = False
        self.results = None
        self.error = None
        self.task = None

    def get_inference_params(self):
        # Start with default config
        params = self.config.copy()
        
        # Override with request params if provided
        for key, value in self.request.dict(exclude_unset=True).items():
            if value is not None:
                params[key] = value
                
        # Ensure output path is set using base_output_path and batch_id
        if not params.get("output_path") or params["output_path"] is None:
            params["output_path"] = os.path.join(self.base_output_path, self.id)
        
        # Make sure the output path exists
        os.makedirs(params["output_path"], exist_ok=True)
            
        return params


async def run_inference_without_asyncio_run(
    requests,
    output_path,
    api_model,
    api_key,
    api_base=None,
    api_proxy=None,
    api_timeout=30.0,
    api_max_retries=5,
    generation_config=None,
    num_workers=8,
    request_limit=100,
    request_limit_period=60,
    max_error_count=100,
    trial_run=False,
    dump_individual_rsp=True,
):
    """Modified version of run_api_inference that works within an existing event loop"""
    
    logger.info(f"num_requests: {len(requests)}, output_path: {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    if dump_individual_rsp:
        os.makedirs(os.path.join(output_path, "responses"), exist_ok=True)
    
    if os.path.exists(os.path.join(output_path, "all_responses.jsonl")):
        os.remove(os.path.join(output_path, "all_responses.jsonl"))
    
    client = OpenAIClient(
        output_path=output_path,
        api_model=api_model,
        api_key=api_key,
        api_base=api_base,
        api_proxy=api_proxy,
        api_timeout=api_timeout,
        api_max_retries=api_max_retries,
        generation_config=generation_config,
        trial_run=trial_run,
        dump_individual_rsp=dump_individual_rsp,
        max_error_count=max_error_count,
    )
    
    # Create a pool and run queries
    pool = AsyncRateLimitThreadPool(num_workers, request_limit, request_limit_period)
    writer_task = asyncio.create_task(client.write_responses_to_file())
    
    try:
        coroutines = [pool._rate_limited_call(client.query, req) for req in requests]
        results = await asyncio.gather(*coroutines)
        
        # Wait for all responses to be written
        await client.response_queue.join()
    finally:
        # Always cancel the writer task
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass
    
    return results


async def run_batch_task(task: BatchTask):
    """Execute a batch task asynchronously."""
    try:
        # Get inference parameters
        params = task.get_inference_params()
        
        # Convert batch request to OpenAI request format
        api_requests = []
        for i, req in enumerate(task.request.requests):
            # Prepare messages using unified processing
            messages, _ = prepare_chat_messages(
                req["principle"],
                req["prompt"],
                req["responses"],
                req.get("mask_responses", True)
            )
            
            api_requests.append({
                "uuid": f"batch_{task.id}_{i}",
                "messages": messages
            })
        
        # Run inference
        await run_inference_without_asyncio_run(
            requests=api_requests,
            **{k: v for k, v in params.items() if k != "requests"}
        )
        
        # Load results
        results_path = os.path.join(params["output_path"], "all_responses.jsonl")
        results = []
        
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result_data = json.loads(line.strip())
                    content = result_data["response"]["generated_text"]
                    
                    # Parse using unified processing
                    reward_result = parse_rewardanything_output(content, None)  # No masking for batch
                    
                    results.append({
                        "thoughts": reward_result.reasoning,
                        "results": {
                            "scores": reward_result.scores,
                            "best-to-worst": reward_result.ranking
                        }
                    })
        
        task.results = results
        task.complete = True
        
    except Exception as e:
        logger.exception(f"Batch task {task.id} failed")
        task.error = str(e)
        task.complete = True


@app.post("/api/rewardanything", response_model=RewardResponse)
async def rewardanything_single(request: RewardRequest):
    """Single RewardAnything evaluation."""
    # Get config from app state
    config = app.state.config
    
    # Prepare messages using unified processing
    messages, masked2real = prepare_chat_messages(
        request.principle,
        request.prompt,
        request.responses,
        request.mask_responses
    )
    
    # Create a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Prepare the request for OpenAI inference
    api_request = {
        "uuid": request_id,
        "messages": messages
    }
    
    # Create temporary output directory
    output_path = os.path.join(app.state.base_output_path, f"single_{request_id}")
    
    try:
        # Run inference
        await run_inference_without_asyncio_run(
            requests=[api_request],
            output_path=output_path,
            api_model=config.get("api_model"),
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            api_proxy=config.get("api_proxy"),
            api_timeout=config.get("api_timeout", 120.0),
            api_max_retries=config.get("api_max_retries", 5),
            generation_config=config.get("generation_config"),
            num_workers=1,
            request_limit=config.get("request_limit", 100),
            request_limit_period=config.get("request_limit_period", 60),
            max_error_count=config.get("max_error_count", 50),
            dump_individual_rsp=False
        )
        
        # Load and parse result
        results_path = os.path.join(output_path, "all_responses.jsonl")
        if not os.path.exists(results_path):
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            result_data = json.loads(f.readline().strip())
        
        # Parse the response content using unified processing
        content = result_data["response"]["generated_text"]
        reward_result = parse_rewardanything_output(content, masked2real)
        
        # Convert to API response format
        response = RewardResponse(
            thoughts=reward_result.reasoning,
            results={
                "scores": reward_result.scores,
                "best-to-worst": reward_result.ranking
            }
        )
        
        # Cleanup temporary directory
        try:
            shutil.rmtree(output_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup directory {output_path}: {cleanup_error}")
        
        return response
    
    except Exception as e:
        # Cleanup on error as well
        try:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup directory {output_path}: {cleanup_error}")
        
        logger.exception("Error processing reward request")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/rewardanything_batch")
async def rewardanything_batch(requests: List[RewardRequest], max_timeout: int = 600):
    """Batch RewardAnything evaluation."""
    if not requests:
        raise HTTPException(status_code=400, detail="No requests provided")
    
    # Get config from app state
    config = app.state.config
    
    # Create a unique batch ID
    batch_id = str(uuid.uuid4())
    batch_output_path = os.path.join(app.state.base_output_path, f"batch_{batch_id}")
    
    try:
        # Prepare all requests using unified processing
        api_requests = []
        masked_mappings = {}  # Store masked->real mappings for each request
        
        for i, req in enumerate(requests):
            request_id = f"{batch_id}_{i}"
            
            # Prepare messages
            messages, masked2real = prepare_chat_messages(
                req.principle,
                req.prompt,
                req.responses,
                req.mask_responses
            )
            
            # Store the mapping for later use
            if masked2real:
                masked_mappings[request_id] = masked2real
            
            api_requests.append({
                "uuid": request_id,
                "messages": messages
            })
        
        # Run batch inference
        dump_individual_rsp = config.get("dump_individual_rsp", True)
        await run_inference_without_asyncio_run(
            requests=api_requests,
            output_path=batch_output_path,
            api_model=config.get("api_model"),
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            api_proxy=config.get("api_proxy"),
            api_timeout=config.get("api_timeout", 120.0),
            api_max_retries=config.get("api_max_retries", 5),
            generation_config=config.get("generation_config"),
            num_workers=config.get("num_workers", 8),
            request_limit=config.get("request_limit", 100),
            request_limit_period=config.get("request_limit_period", 60),
            max_error_count=config.get("max_error_count", 50),
            dump_individual_rsp=dump_individual_rsp
        )
        
        # Load and process the results
        results_path = os.path.join(batch_output_path, "all_responses.jsonl")
        if not os.path.exists(results_path):
            raise HTTPException(status_code=500, detail="Failed to generate responses")
        
        # Parse each response using unified processing
        results = []
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                result_data = json.loads(line.strip())
                request_id = result_data["request"]["uuid"]
                content = result_data["response"]["generated_text"]
                
                # Get the masked->real mapping for this request
                masked2real = masked_mappings.get(request_id)
                
                # Parse the response using unified processing
                reward_result = parse_rewardanything_output(content, masked2real)
                
                # Convert to API response format
                response = RewardResponse(
                    thoughts=reward_result.reasoning,
                    results={
                        "scores": reward_result.scores,
                        "best-to-worst": reward_result.ranking
                    }
                )
                results.append(response)
        
        # Cleanup temporary directory if dump_individual_rsp is enabled
        if dump_individual_rsp:
            try:
                shutil.rmtree(batch_output_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup directory {batch_output_path}: {cleanup_error}")
        
        return results
        
    except Exception as e:
        # Cleanup on error as well if dump_individual_rsp is enabled
        dump_individual_rsp = config.get("dump_individual_rsp", True)
        if dump_individual_rsp:
            try:
                if os.path.exists(batch_output_path):
                    shutil.rmtree(batch_output_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup directory {batch_output_path}: {cleanup_error}")
        
        logger.exception(f"Error processing batch request: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/batch_request", response_model=BatchRequestResponse)
async def create_batch_request(request: BatchRequest, background_tasks: BackgroundTasks):
    """Create a new batch request for asynchronous processing."""
    task = BatchTask(request, app.state.config, app.state.base_output_path)
    batch_tasks[task.id] = task
    
    # Start the task in the background
    background_tasks.add_task(run_batch_task, task)
    
    return BatchRequestResponse(
        batch_request_id=task.id,
        requests=len(request.requests),
        start_time=task.start_time
    )


@app.get("/api/batch_request/{batch_request_id}")
async def get_batch_request(batch_request_id: str, max_timeout: int = 600):
    """Get the results of a batch request."""
    if batch_request_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="Batch request not found")
    
    task = batch_tasks[batch_request_id]
    
    start_time = time.time()
    while not task.complete and (time.time() - start_time) < max_timeout:
        await asyncio.sleep(1)
    
    if not task.complete:
        raise HTTPException(status_code=408, detail="Request timed out")
    
    if task.error:
        raise HTTPException(status_code=500, detail=f"Task failed: {task.error}")
    
    return task.results


@app.post("/api/clear_tasks")
async def clear_tasks():
    batch_tasks.clear()
    return {"message": "All tasks cleared"}


@app.get("/api/list_tasks")
async def list_tasks():
    return {
        task_id: {
            "start_time": task.start_time,
            "complete": task.complete,
            "has_error": task.error is not None,
            "num_requests": len(task.request.requests),
            "output_path": os.path.join(task.base_output_path, task.id)
        }
        for task_id, task in batch_tasks.items()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RewardAnything API"}


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def create_app(config: Dict, base_output_path: str) -> FastAPI:
    """Create FastAPI app with configuration"""
    # Inject config into the app
    app.state.config = config
    app.state.base_output_path = base_output_path
    return app


def main():
    parser = argparse.ArgumentParser(description="RewardAnything API Server")
    parser.add_argument("-c", "--config", required=True, help="Path to configuration file")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--base-output-path", default="./outputs", 
                       help="Base directory for storing batch outputs")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get absolute path for base output path
    base_output_path = os.path.abspath(args.base_output_path)
    
    # Create the output directory if it doesn't exist
    os.makedirs(base_output_path, exist_ok=True)
    
    # Create app with config
    app = create_app(config, base_output_path)
    
    logger.info(f"Server starting with base output path: {base_output_path}")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 