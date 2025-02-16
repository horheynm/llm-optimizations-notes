import time
import torch
import queue
import threading
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MAX_TIME_WAIT = 0.05  
start_time = time.time()  


class DynamicBatcher:
    """Continous batching scheduler"""
    
    def __init__(self, model_name="distilgpt2", max_batch_size=4, max_seq_len=64, max_time=MAX_TIME_WAIT):
        """
        :param model_name: HF stub
        :param max_batch_size: Max number of requests to combine in one batch
        :param max_seq_len: Max sequence length to generate
        :param max_time: Max time in seconds to wait for requests before batching and carrying out inference.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model: {model_name} onto {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_time = max_time
        self.request_queue = queue.Queue() # thread safe

        self.running = True
        self.lock = threading.Lock()

        # start the batching thread
        self.batch_thread = threading.Thread(target=self._batch_loop)
        self.batch_thread.start()

    def add_request(self, text_prompt):
        """
        Adds a new text request to the queue.

        :param text_prompt: Input text to generate from.
        :return: A unique request ID (timestamp-based).
        """
        req_time = time.time() 
        relative_arrival = req_time - start_time  # convert to T=0 for reference

        encoded_inputs = self.tokenizer(text_prompt, return_tensors="pt")
        
        # remove the extra batch dimension
        input_ids = encoded_inputs["input_ids"].squeeze(0)            # shape => (seq_len,)
        attention_mask = encoded_inputs["attention_mask"].squeeze(0)  # shape => (seq_len,)

        self.request_queue.put((req_time, input_ids, attention_mask, text_prompt))

        print(f"[T={relative_arrival:.4f}] Prompt arrived: '{text_prompt}'")
        return req_time

    def _batch_loop(self):
        """
        Continuously processes requests in batches.
        Gathers requests from the queue until either:
            we reach max_batch_size 
            max_time passes

        Then processes them in a single batch.
        """
        print("\nGenerating using dynamic batching....\n")
        while self.running:
            batch_requests = []
            gather_start_time = time.time()

            # 1. add to queue max_batch_size requests or until max_time passes
            while (
                (time.time() - gather_start_time) < self.max_time
                and (len(batch_requests) < self.max_batch_size)
            ):
                try:
                    req = self.request_queue.get(timeout=0.01)
                    batch_requests.append(req)
                except queue.Empty:
                    pass

                # full batch
                if len(batch_requests) >= self.max_batch_size:
                    break

            if not batch_requests:
                continue

            batch_time = time.time()
            relative_batch_time = batch_time - start_time

            request_times, input_tensors, attention_masks, text_prompts = zip(*batch_requests)

            # 2. Pad input_ids => shape (batch_size, max_seq_len_in_batch)
            batch_input = torch.nn.utils.rnn.pad_sequence(
                input_tensors, 
                batch_first=True, 
                padding_value=self.tokenizer.pad_token_id,
                padding_side='left',
            ).to(self.device)

            # 3. Pad attention_masks => shape (batch_size, max_seq_len_in_batch)
            batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_masks, 
                batch_first=True, 
                padding_value=0,
                padding_side='left',
            ).to(self.device)

            print(f"\n[T={relative_batch_time:.4f}s] Batch Start::Processing {len(batch_requests)} request(s)...\n")

            # 4. Run model.generate on the entire batch
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_input,
                    attention_mask=batch_attention_mask,
                    max_length=self.max_seq_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )

            # 5. Decode
            generated_texts = [
                self.tokenizer.decode(out, skip_special_tokens=True)
                for out in outputs
            ]

            for idx, gen_text in enumerate(generated_texts):
                req_time_abs = request_times[idx]
                req_time_rel = req_time_abs - start_time
                prompt_str = text_prompts[idx]
                print(f"[T={req_time_rel:.4f}s]::Prompt '{prompt_str}'")
                print(f"                         Generated text:\n   {gen_text}\n")

    def stop(self):
        """Stops the batch processing thread."""
        self.running = False
        self.batch_thread.join()


def get_poisson_delay(mean_interarrival=(MAX_TIME_WAIT / 3)):
    """
    Returns a single random sample from an exponential distribution
    (i.e., the inter-arrival time in a Poisson process).
    
    :param mean_interarrival: The average time between arrivals.
    :return: A float representing one random inter-arrival delay.
    """
    return random.expovariate(1.0 / mean_interarrival)

def load_prompts_from_file(filename="prompts.txt"):
    """
    Generator that yields prompts line by line from the given file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


if __name__ == "__main__":
    batcher = DynamicBatcher(
        # model_name="EleutherAI/gpt-neo-1.3B", # just make sure dynamic batching works, generation coherence dont matter
        max_batch_size=4,
        max_seq_len=64,
        max_time=MAX_TIME_WAIT
    )
    try:
        for prompt in load_prompts_from_file(os.path.abspath("batching/prompts.txt")):
            batcher.add_request(prompt)
            time.sleep(get_poisson_delay()) # sample from a poisson distrubution

        while batcher.request_queue.qsize() != 0:
            time.sleep(1)
    # except Exception:
    #     batcher.stop()
    finally:
        batcher.stop()
        print("Batcher terminated")
