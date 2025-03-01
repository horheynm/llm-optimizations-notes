from typing import List, Tuple
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
            # print(f"step {i} input: {result}")
            
            input_ids = self.encode(result)
            
            next_token_probs = self.get_next_token_prob(input_ids)
            
            id = torch.argmax(next_token_probs, dim=-1).item()
            
            result += self.decode(id)
            
            predictions.append(next_token_probs[id])
            
        print("[Greedyl]: ", result)
        

class BeamSearchSampler(Sampler):
    def __init__(self, model_name: str = "gpt2-medium", num_beams: int = 3):
        super().__init__(model_name=model_name)
        self.num_beams = num_beams

    def __call__(self, prompt: str, max_length: int = 10):
        input_ids = self.encode(prompt)
        
        output_ids = self.model.generate(
            input_ids,
            num_beams=self.num_beams,
            max_length=input_ids.shape[1] + max_length,
            # early_stopping=True,
            # no_repeat_ngram_size=2,  # helps reduce repeating n-grams
        )
        
        print("[BeamSearchModel] Final:", self.decode(output_ids[0]))
        


class BeamSearchSamplerManual(Sampler):
    def __init__(self, model_name: str = "gpt2-medium", num_beams: int = 3):
        super().__init__(model_name=model_name)
        self.num_beams = num_beams

    def __call__(self, prompt: str, max_length: int = 10):
        """
        Manual beam search decoding:
         1. Start with a single beam containing the prompt (input_ids).
         2. For each beam, get the top beam candidates for the next token.
         3. Merge all candidates, then take the top self.num_beams across them.
         4. Repeat until max_length steps are reached.
        """
        # Encode the initial prompt
        input_ids = self.encode(prompt)  # shape: [1, seq_len]
        
        # Each entry in `beams` is a tuple: (token_ids, cumulative_log_prob).
        # We'll start with just one beam containing the prompt.
        beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        
        # Expand the sequence up to max_length tokens (beyond the initial prompt length)
        for step in range(max_length):
            new_beams = []
            
            # For every existing beam, try all expansions
            for seq, seq_score in beams:
                next_token_logits = self.get_next_token_prob(seq)
                
                # Convert logits -> log probabilities for better numeric stability when summing
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Get top `num_beams` token indices & log probabilities
                topk_log_probs, topk_indices = torch.topk(next_token_log_probs, self.num_beams)
                
                for log_prob, token_id in zip(topk_log_probs, topk_indices):
                    # Create new sequence by appending this token
                    expanded_seq = torch.cat([seq, token_id.view(1,1)], dim=1)
                    
                    # Update cumulative log probability
                    new_score = seq_score + log_prob.item()
                    
                    new_beams.append((expanded_seq, new_score))
            
            # Out of the expanded candidate set, keep only top `num_beams` beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[: self.num_beams]
        
        # The best beam will be the first after sorting by score (descending)
        best_seq, best_score = beams[0]
        
        # Decode and return
        result = self.decode(best_seq[0])  # best_seq is shape [1, seq_len]
        print("[BeamSearchManual] Final:", result)
        return result

if __name__ == "__main__":
    prompt = "Hello my name is George and I am working on"
    max_length = 64
    
    print("=== Greedy ===")
    greedy = GreedySampler()
    greedy(prompt, max_length=max_length)

    print("\n=== Beam Search in Model ===")
    beam_search = BeamSearchSampler(num_beams=3)
    beam_search(prompt, max_length=max_length)
    
    
    print("\n=== Beam Search Manual ===")
    beam_search = BeamSearchSamplerManual(num_beams=3)
    beam_search(prompt, max_length=max_length)
    
    """
    Manual and Model beam search differ bc there will be some differences in the backend.
    length penalty, repetition penalty, diversity penanlty, no repeat ngram size, early stopping etc..
    
    """




"""
=== Greedy ===
[Greedyl]:  Hello my name is George and I am working on a project called "The World's First Real-Time Video Game". I am a software engineer and I am working on a game called "The World's First Real-Time Video Game". I am a software engineer and I am working on a game called "The World's First Real-Time Video Game".

=== Beam Search in Model ===
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[BeamSearchModel] Final: Hello my name is George and I am working on a project that I am very passionate about. I have been working on this project for over a year now and I am very excited to be able to share it with the world.

I have been working on this project for over a year now and I am very excited to be able to share it with the world

=== Beam Search Manual ===
[BeamSearchManual] Final: Hello my name is George and I am working on a project that I am very passionate about. I have been working on this project for over a year now and I am very excited to be able to share it with the world.

I have been working on this project for over a year now and I am very excited to be able to share it with the world

"""