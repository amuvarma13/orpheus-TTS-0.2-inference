import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import threading
import queue
import numpy as np
from snac import SNAC

class EngineClass:
    def __init__(self, model_name, dtype=torch.bfloat16, snac_model_path="hubertsiuzdak/snac_24khz"):
        self.model_name = model_name
        self.dtype = dtype
        self.engine = self._setup_engine()
        self._setup_snac_model(snac_model_path)
        
    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    
    def _setup_snac_model(self, snac_model_path):
        self.snac_model = SNAC.from_pretrained(snac_model_path).eval()
        self.snac_device = "cuda"
        self.snac_model = self.snac_model.to(self.snac_device)
    
    def _format_prompt(self, prompt):
        return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
    
    def _convert_to_audio(self, multiframe, count):
        frames = []
        if len(multiframe) < 7:
            return
        
        codes_0 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.snac_device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]

        for j in range(num_frames):
            i = 7*j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)])

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
            
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
            return

        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
        
        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

    def _turn_token_into_id(self, token_string, index):
        token_string = token_string.strip()
        last_token_start = token_string.rfind("<custom_token_")
        
        if last_token_start == -1:
            print("No token found in the string")
            return None
        
        last_token = token_string[last_token_start:]
        
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except ValueError:
                print(f"Failed to convert '{number_str}' to integer")
                return None
        else:
            print("The token format is incorrect:", repr(last_token))
            return None
    
    async def _tokens_decoder(self, token_gen):
        buffer = []
        count = 0
        async for token_sim in token_gen:       
            token = self._turn_token_into_id(token_sim, count)
            if token is None:
                print("*")
            else:
                if token > 0:
                    buffer.append(token)
                    count += 1

                    if count % 7 == 0 and count > 27:
                        buffer_to_proc = buffer[-28:]
                        audio_samples = self._convert_to_audio(buffer_to_proc, count)
                        if audio_samples is not None:
                            yield audio_samples
    
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

        # First, get the tokens from vLLM
        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(formatted_prompt, sampling_params, request_id=request_id):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        # Now pass these tokens through the decoder to get audio
        audio_queue = queue.Queue()

        async def async_token_gen():
            while True:
                token = token_queue.get()
                if token is None:
                    break
                yield token

        async def audio_producer():
            async for audio_chunk in self._tokens_decoder(async_token_gen()):
                audio_queue.put(audio_chunk)
            audio_queue.put(None)  # Sentinel

        def run_audio_async():
            asyncio.run(audio_producer())

        audio_thread = threading.Thread(target=run_audio_async)
        audio_thread.start()

        # Yield audio chunks to the client
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            yield audio

        thread.join()
        audio_thread.join()
    def generate_tokens_sync(self, 
                            prompt, 
                            request_id="req-001", 
                            temperature=0.6, 
                            top_p=0.8, 
                            max_tokens=1200, 
                            stop_token_ids=[49158], 
                            repetition_penalty=1.3):
        """
        Synchronous wrapper for generate_tokens_async.
        This function wraps the asynchronous generator using a dedicated event loop
        in a separate thread and a thread-safe queue.
        """
        import asyncio
        import threading
        import queue

        audio_queue = queue.Queue()
        loop = asyncio.new_event_loop()

        async def async_runner():
            async for audio_chunk in self.generate_tokens_async(
                prompt=prompt,
                request_id=request_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                repetition_penalty=repetition_penalty
            ):
                audio_queue.put(audio_chunk)
            # Signal the end of the generator with a sentinel value
            audio_queue.put(None)

        def runner(loop):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(async_runner())
            loop.close()

        thread = threading.Thread(target=runner, args=(loop,))
        thread.start()

        # Yield audio chunks synchronously as they become available.
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            yield audio

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
        
        # Create async generator for tokens
        async def token_generator():
            async for result in self.engine.generate(formatted_prompt, sampling_params, request_id=request_id):
                yield result.outputs[0].text
        
        # Pass tokens through the decoder and yield audio
        async for audio_chunk in self._tokens_decoder(token_generator()):
            yield audio_chunk