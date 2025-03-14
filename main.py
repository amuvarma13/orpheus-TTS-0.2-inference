import asyncio
import os
import gc
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from tokens_decoder import tokens_decoder
import uuid

class MinimalLLM:
    def __init__(self, model_name="canopylabs/orpheus-tts-0.1-primary"):
        self.model_name = model_name
        self.start_token = torch.tensor([[128259]], dtype=torch.int64)
        self.end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        self.sampling_params = SamplingParams(
            temperature=0.9, 
            top_p=0.6, 
            max_tokens=2000, 
            repetition_penalty=1.1, 
            stop_token_ids=[128258]
        )
        self.initialize_model()

    def initialize_model(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        engine_args = AsyncEngineArgs(
            model=self.model_name, 
            dtype=torch.float16,
            gpu_memory_utilization=0.8
        )
        self.model = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def process_prompt(self, prompt: str) -> str:
        prompt = prompt + " " + "<zac>"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        modified_input_ids = torch.cat([self.start_token, input_ids, self.end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(modified_input_ids[0].tolist())
        return prompt_string



    async def generate_tokens(self, prompt: str):
        prompt_string = self.process_prompt(prompt)
        previous_text = ""
        req_id = str(uuid.uuid4())  # Generate a unique request id
        async for result in self.model.generate(prompt_string, self.sampling_params, request_id=req_id):
            new_text = result.outputs[0].text[len(previous_text):]
            previous_text = result.outputs[0].text
            if new_text:
                yield new_text


def main():
    prompt = "Hello world, this is a minimal test"
    minimal_llm = MinimalLLM()

    async def raw_token_generator():
        async for text in minimal_llm.generate_tokens(prompt):
            yield text

    for decoded_output in tokens_decoder(raw_token_generator()):
        print(decoded_output, end="", flush=True)

main()

