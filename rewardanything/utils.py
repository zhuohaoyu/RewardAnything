import asyncio
import json
import os
import argparse
import logging
import time
import codecs
import traceback

from typing import Optional, List, Dict, Union, Any
from random import choice
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI, APIStatusError


class OpenAIClient:
    def __init__(
        self,
        output_path: str,
        api_model: str,
        api_key: Union[str, List[str]],
        api_base: Optional[Union[str, List[str]]] = None,
        api_proxy: Optional[Union[str, List[str]]] = None,
        api_timeout: Optional[float] = 60.0,
        api_max_retries: Optional[int] = 5,
        generation_config: Optional[Dict] = None,
        max_error_count: Optional[int] = 100,
        trial_run=False,
        dump_individual_rsp=True,
    ):
        self.output_path = output_path
        self.trial_run = trial_run
        self.max_error_count = max_error_count
        self.total_errors = 0
        self.logger = logging.getLogger(__name__)

        if isinstance(api_key, str):
            api_key = [api_key]

        if api_base is None:
            api_base = ["https://api.openai.com/v1"] * len(api_key)
        elif isinstance(api_base, str):
            api_base = [api_base] * len(api_key)

        assert len(api_key) == len(
            api_base
        ), "Number of api_key and api_base must match"

        if api_proxy is not None:
            if isinstance(api_proxy, str):
                api_proxy = [api_proxy] * len(api_key)
            else:
                assert len(api_proxy) == len(
                    api_key
                ), "Number of api_key and api_proxy must match"
            self.clients = [
                AsyncOpenAI(
                    api_key=key,
                    base_url=api_base,
                    timeout=api_timeout,
                    max_retries=api_max_retries,
                )
                for key, api_base, proxy in zip(
                    api_key, api_base, api_proxy
                )
            ]
        else:
            self.clients = [    
                AsyncOpenAI(
                    api_key=key,
                    base_url=api_base,
                    timeout=api_timeout,
                    max_retries=api_max_retries,
                )
                for key, api_base in zip(api_key, api_base)
            ]

        self.model = api_model

        self.response_queue = asyncio.Queue()
        self.dump_individual_rsp = dump_individual_rsp

        if generation_config is None:
            self.generation_config = {
                "frequency_penalty": 0,
                "max_tokens": 100,
                "n": 1,
                "presence_penalty": 0,
                "response_format": {"type": "text"},
                "seed": 42,
                "stream": False,
                "temperature": 0.0,
            }
        else:
            self.generation_config = generation_config

        if dump_individual_rsp:
            os.makedirs(os.path.join(self.output_path, "responses"), exist_ok=True)

    async def query(
        self,
        request,
        num_retries=3,
    ):
        if isinstance(request, dict):
            request_dict = request
        else:
            request_dict = request.__dict__

        assert "messages" in request_dict, "messages must be provided in request"
        assert "uuid" in request_dict, "uuid must be provided in request"

        if self.dump_individual_rsp:
            save_path = os.path.join(
                self.output_path, "responses", f'{request_dict["uuid"]}.json'
            )

            if os.path.exists(save_path) and not self.trial_run:
                with codecs.open(save_path) as f:
                    rsp_content = json.load(f)
                await self.response_queue.put(rsp_content)
                return f"Skipping {save_path}"

        if "generation_config" in request_dict and isinstance(
            request_dict["generation_config"], dict
        ):
            generation_config = self.generation_config.copy()
            generation_config.update(request_dict["generation_config"])
        else:
            generation_config = self.generation_config

        response = None
        while num_retries > 0:
            num_retries -= 1
            try:
                client = choice(self.clients)
                response = await client.chat.completions.create(
                    messages=request_dict["messages"],
                    model=self.model,
                    **generation_config,
                )
                response = response.model_dump()
                break
            except APIStatusError as e:
                if self.max_error_count > self.total_errors:
                    self.total_errors += 1
                    self.logger.warning(
                        f"OpenAI APIStatusError: {e}, total errors: {self.total_errors}, sleeping..."
                    )
                    await asyncio.sleep(1.0)
                else:
                    self.logger.error(
                        f"OpenAI APIStatusError: {e}, max_error_count reached, exiting..."
                    )
                    raise e
            except:
                print(traceback.format_exc())

        if response is None:
            raise Exception("Empty response from remote OpenAI API")

        try:
            response["generated_text"] = response["choices"][0]["message"]["content"]
        except:
            print(traceback.format_exc())
            print(response)
            raise Exception("Empty response from remote OpenAI API")

        if self.dump_individual_rsp:
            with codecs.open(save_path, "w") as f:
                json.dump(
                    {"request": request_dict, "response": response},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        await self.response_queue.put({"request": request_dict, "response": response})

        return response["choices"][0]["message"]["content"]

    async def write_responses_to_file(self):
        save_path = os.path.join(self.output_path, "all_responses.jsonl")
        while True:
            response = await self.response_queue.get()
            with codecs.open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")
            self.response_queue.task_done()


class AsyncRateLimitThreadPool:
    def __init__(self, num_workers, num_requests, period):
        self.num_workers = num_workers
        self.num_requests = num_requests
        self.loop = asyncio.get_event_loop()
        self.semaphore = asyncio.Semaphore(num_workers)
        self.last_call_time = time.time()
        self.call_count = 0
        self.period = period

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def _rate_limited_call(self, func, *args, **kwargs):
        # Limit the number of calls to func per minute
        elapsed_time = time.time() - self.last_call_time
        if elapsed_time < self.period:
            self.call_count += 1
            if self.call_count > self.num_requests:
                sleep_time = self.period - elapsed_time
                # logging.info("Sleeping for {} seconds".format(sleep_time))
                await asyncio.sleep(sleep_time)
                self.call_count = 0
                self.last_call_time = time.time()

        # Acquire a semaphore permit before calling func
        async with self.semaphore:
            result = await func(*args, **kwargs)

        return result

    async def map(self, func, *args_list):
        coroutines = [self._rate_limited_call(func, *args) for args in zip(*args_list)]

        # Use tqdm progress bar with coroutines
        results = []
        for coroutine in atqdm.as_completed(coroutines):
            result = await coroutine
            results.append(result)

        return results


async def run_pool(api, requests, num_workers, num_requests, period):
    pool = AsyncRateLimitThreadPool(num_workers, num_requests, period)
    writer_task = asyncio.create_task(api.write_responses_to_file())

    results = await pool.map(api.query, requests)
    await api.response_queue.join()  # Ensure all responses are written
    writer_task.cancel()

    return results


def run_api_inference(
    requests: Union[
        List[Dict], Any
    ],  # can List[Dict] or list of any object with __dict__ attribute
    output_path: str,  # path to save responses
    api_model: str,  # openai model name
    api_key: Union[str, List[str]],
    api_base: Optional[Union[str, List[str]]] = None,
    api_proxy: Optional[Union[str, List[str]]] = None,
    api_timeout: Optional[float] = 30.0,
    api_max_retries: Optional[int] = 5,
    generation_config: Optional[Dict] = None,
    num_workers: Optional[int] = 8,
    request_limit: Optional[int] = 100,
    request_limit_period: Optional[int] = 60,
    max_error_count: Optional[int] = 100,
    trial_run=False,
    dump_individual_rsp=True,
):
    logging.getLogger(__name__).info(
        f"num_requests: {len(requests)}, output_path: {output_path}"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

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
    )

    try:
        asyncio.run(
            run_pool(
                client,
                requests,
                num_workers=num_workers,
                num_requests=request_limit,
                period=request_limit_period,
            )
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupt received! Closing...")
