import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import threading
import queue

# Setup engine arguments with the target model and desired settings.
engine_args = AsyncEngineArgs(
    model="amuvarma/135m-tts-tune-checkpoint-1300-of-1300",
    dtype=torch.float16,
    gpu_memory_utilization=0.8
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Define sampling parameters for generation.
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=50  # Adjust max_tokens as needed.
)

def generate_tokens_sync(prompt, request_id="req-001"):
    """
    A synchronous generator that yields tokens from the asynchronous vLLM engine.
    Internally, it runs an async producer in a background thread and uses a queue
    to pass tokens to the synchronous context.
    """
    token_queue = queue.Queue()

    async def async_producer():
        # Generate tokens asynchronously.
        async for result in engine.generate(prompt, sampling_params, request_id=request_id):
            token_queue.put(result.outputs[0].text)
        token_queue.put(None)  # Sentinel value to signal completion.

    def run_async():
        # Run the async_producer in its own event loop.
        asyncio.run(async_producer())

    # Start the async producer in a background thread.
    thread = threading.Thread(target=run_async)
    thread.start()

    # Yield tokens synchronously as they are placed on the queue.
    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token

    thread.join()

# Now the user can access generated tokens using a normal for loop:
prompt = "Hello, this is a test prompt."
for token in generate_tokens_sync(prompt):
    print(token, end="")
