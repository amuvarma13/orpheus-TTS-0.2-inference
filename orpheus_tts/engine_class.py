import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import threading
import queue

class EngineClass:
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = model_name
        self.dtype = dtype
        self.engine = self._setup_engine()
        
    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    
    def _format_prompt(self, prompt):
        return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
    
    def generate_tokens(self, 
                        prompt, 
                        request_id="req-001", 
                        temperature=0.6, 
                        top_p=0.8, 
                        max_tokens=1200, 
                        stop_token_ids=[49158], 
                        repetition_penalty=1.3):
        formatted_prompt = self._format_prompt(prompt)
        print(formatted_prompt)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(formatted_prompt, sampling_params, request_id=request_id):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

    async def generate_tokens_async(self, 
                                    prompt, 
                                    request_id="req-001", 
                                    temperature=0.6, 
                                    top_p=0.8, 
                                    max_tokens=1200, 
                                    stop_token_ids=[49158], 
                                    repetition_penalty=1.3):
        formatted_prompt = self._format_prompt(prompt)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )
        
        async for result in self.engine.generate(formatted_prompt, sampling_params, request_id=request_id):
            yield result.outputs[0].text