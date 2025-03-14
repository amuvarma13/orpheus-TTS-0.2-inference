import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import threading
import queue
import tokens_decoder  # This module should define an async generator called tokens_decoder

# ------------------ Setup the vLLM Engine ------------------ #
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

# ------------------ Synchronous Token Generator ------------------ #
def generate_tokens_sync(prompt, request_id="req-001"):
    """
    Synchronous generator that yields tokens produced asynchronously by the vLLM engine.
    It runs an async producer in a background thread and uses a queue to pass tokens to the
    synchronous caller.
    """
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

# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """
    Wraps the asynchronous tokens_decoder (from tokens_decoder module) so that it accepts a
    synchronous token generator and yields decoded audio chunks synchronously.
    """
    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator that processes tokens.
        async for audio_chunk in tokens_decoder.tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()

# ------------------ Synchronous Usage Example ------------------ #
prompt = "Hello, this is a test prompt."

# First, generate tokens synchronously from the prompt.
syn_tokens = generate_tokens_sync(prompt)

# Then, decode those tokens to get audio chunks (synchronously).
for audio_chunk in tokens_decoder_sync(syn_tokens):
    # Here we "log" the audio stream.
    # For demonstration, we'll print the length of each audio chunk.
    # In a real application you might write the audio to a file or process it further.
    print("Audio chunk received of length:", len(audio_chunk))
