import time
import queue
import asyncio
import torch
import gc
import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer


class LLMModelManager:
    def __init__(self, model_name="canopylabs/orpheus-tts-0.1-primary"):
        self.model_name = model_name
        self.model = None
        self.tokeniser = None
        self.start_token = torch.tensor([[128259]], dtype=torch.int64)
        self.end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        self.sampling_params = SamplingParams(
            temperature=0.9, 
            top_p=0.6, 
            max_tokens=2000, 
            repetition_penalty=1.1, 
            stop_token_ids=[128258]
        )
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.initialize_model()

    def initialize_model(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name, 
            dtype=torch.float16,
            gpu_memory_utilization=0.8
        )
        self.model = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)

    def process_prompt(self, prompt):
        prompt = prompt + " " + "<zac>"
        input_ids = self.tokeniser(prompt, return_tensors="pt").input_ids
        modified_input_ids = torch.cat([self.start_token, input_ids, self.end_tokens], dim=1)
        iids_string = self.tokeniser.decode(modified_input_ids[0].tolist())
        # Return the processed prompt string (token count not used here)
        return iids_string

    async def process_request(self, prompt, token_queue, request_id, attempt):
        try:
            prompt_string = self.process_prompt(prompt)
            results_generator = self.model.generate(prompt_string, self.sampling_params, request_id=request_id)
            previous_text = ""
            async for request_output in results_generator:
                text = request_output.outputs[0].text
                new_text = text[len(previous_text):]
                previous_text = text
                if new_text:
                    token_queue.put(new_text)
            token_queue.put(None)  # Signal completion
        except Exception:
            token_queue.put("ERROR")

def main():
    prompt = "Hello world"
    token_queue = queue.Queue()
    request_id = "example"
    attempt = 1

    asyncio.run(manager.process_request(prompt, token_queue, request_id, attempt))

    generated_text = ""
    while True:
        token = token_queue.get()
        if token is None or token == "ERROR":
            break
        generated_text += token

    print("Generated text:")
    print(generated_text)

if __name__ == '__main__':
    manager = LLMModelManager()
    main()
