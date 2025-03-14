import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import threading
import queue




# ------------------ Setup the vLLM Engine ------------------ #
engine_args = AsyncEngineArgs(
    model="amuvarma/135m-tts-tune-checkpoint-1300-of-1300",
    dtype=torch.bfloat16,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)




def format_prompt(prompt):
    formatted_prompt = f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
    return formatted_prompt


def generate_tokens_sync(prompt, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids = [49158], repetition_penalty=1.3):
    prompt = format_prompt(prompt)
    print(prompt)
    sampling_params = SamplingParams(
      temperature=temperature,
      top_p=top_p,
      max_tokens=max_tokens,  # Adjust max_tokens as needed.
      stop_token_ids = stop_token_ids, 
      repetition_penalty=repetition_penalty, 
    )

    token_queue = queue.Queue()

    async def async_producer():
        async for result in engine.generate(prompt, sampling_params, request_id=request_id):
            # Place each token text into the queue.
            token_queue.put(result.outputs[0].text)
        token_queue.put(None)  # Sentinel to indicate completion.

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









