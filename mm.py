import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def main():
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
    
    prompt = "Hello, this is a test prompt."
    request_id = "req-001"  # A unique identifier for this request.
    
    # Generate tokens asynchronously and print the output.
    async for result in engine.generate(prompt, sampling_params, request_id=request_id):
         print(result.outputs[0].text, end="")

# Run the async generation process.
asyncio.run(main())
