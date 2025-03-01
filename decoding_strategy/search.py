from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Sampler:
    def __init__(self, model_name: str = "gpt2-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
    def encode(self, text: str):
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids)

    def get_next_token_prob(self, input_ids: torch.Tensor):
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits # [batch, seq_len, vocab_size]
        logits = logits[0, -1, :]
        return logits


class GreedySampler(Sampler):
    def __call__(self, prompt: str, max_length: int = 10):
        predictions = []
        result = prompt
        
        for i in range(max_length):
            print(f"step {i} input: {result}")
            
            input_ids = self.encode(result)
            
            next_token_probs = self.get_next_token_prob(input_ids)
            
            id = torch.argmax(next_token_probs, dim=-1).item()
            
            result += self.decode(id)
            
            predictions.append(next_token_probs[id])
            
        print(result)


if __name__ == "__main__":
    greedy = GreedySampler()
    breakpoint()
    prompt = "Hello my name is "

    greedy(prompt)

