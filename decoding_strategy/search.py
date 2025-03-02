from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



class Sampler:
    def __init__(self, model_name: str = "gpt2-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure GPT-2 can pad using the eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def encode(self, texts):
        """
        texts: either a single string or a list of strings
        returns (input_ids, attention_mask), each shape [batch_size, seq_len]
        """
        if isinstance(texts, str):
            texts = [texts]  # wrap single string

        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        return input_ids, attention_mask
        
    def decode(self, ids: torch.Tensor):
        """
        Takes a batch of sequences and decodes each one into a string.
        """
        return [self.tokenizer.decode(row, skip_special_tokens=True) for row in ids]

    def get_next_token_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids shape: [batch_size, seq_len]
        attention_mask shape: [batch_size, seq_len]
        returns: logits shape [batch_size, vocab_size] for the *last* position
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # outputs.logits: [batch_size, seq_len, vocab_size]
            logits = outputs.logits[:, -1, :]  # shape [batch_size, vocab_size]
        return logits


class GreedySampler(Sampler):
    """
    Picks highest probable token without considering diversity - no penalty
    """
    def __call__(self, prompt: List[str], max_length: int = 10):
        result = prompt
        input_ids, attntion_mask = self.encode(result)  # shapes: [batch_size, seq_len]

        for _ in range(max_length):
            next_token_logits = self.get_next_token_logits(input_ids, attntion_mask)  # [batch_size, vocab_size]
            next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)     # [batch_size, 1]

            # Append the new token to input_ids
            input_ids = torch.cat([input_ids, next_token_ids], dim=1)                # [batch_size, seq_len+1]

            # Also append a '1' to attntion_mask for each newly added token
            new_mask = torch.ones(
                (attntion_mask.shape[0], 1),
                dtype=attntion_mask.dtype,
                device=attntion_mask.device,
            )
            attntion_mask = torch.cat([attntion_mask, new_mask], dim=1)             # [batch_size, seq_len+1]

        # Decode and print results
        for i, input_id in enumerate(input_ids):
            print(f"[Greedy, batch{i}]: ", self.decode(input_id))

        
        
class TopKSampler(Sampler):
    """Select one sample from k selected, where p of being selected is ki / sum(k)"""
    def __init__(self, model_name: str = "gpt2-medium", k: int = 5, temperature: float = 1.0):
        super().__init__(model_name)
        self.k = k
        self.temperatrure = temperature

    def __call__(self, prompt: str, max_length: int = 5):
        result = prompt 
        input_ids = self.encode(result)

        for _ in range(max_length):
            next_token_logits = self.get_next_token_logits(input_ids)
            next_token_logits = next_token_logits / self.temperatrure
            top_k_logits, top_k_indices = torch.topk(next_token_logits, self.k)
            top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

            chosen_index = torch.multinomial(top_k_probs, num_samples=1)
            token_id = top_k_indices[chosen_index].view(-1, 1)
            input_ids = torch.cat((input_ids, token_id), dim=1)
        
        print("[TopK]: ", self.decode(input_ids[0]))


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
            early_stopping=True,
            no_repeat_ngram_size=2,  # helps reduce repeating n-grams
        )
        
        print("[BeamSearchModel] Final:", self.decode(output_ids[0]))
        

# class BeamSearchSamplerManual(Sampler):
#     def __init__(self, model_name: str = "gpt2-medium", num_beams: int = 3):
#         super().__init__(model_name=model_name)
#         self.num_beams = num_beams

#     def __call__(self, prompt: str, max_length: int = 10):
#         """
#         Manual beam search decoding:
#          1. Start with a single beam containing the prompt (input_ids).
#          2. For each beam, get the top beam candidates for the next token.
#          3. Merge all candidates, then take the top self.num_beams across them.
#          4. Repeat until max_length steps are reached.
#         """
#         # Encode the initial prompt
#         input_ids = self.encode(prompt)  # shape: [1, seq_len]
        
#         # Each entry in `beams` is a tuple: (token_ids, cumulative_log_prob).
#         # We'll start with just one beam containing the prompt.
#         beams: List[Tuple[torch.Tensor, float]] = [(input_ids, 0.0)]
        
#         # Expand the sequence up to max_length tokens (beyond the initial prompt length)
#         for step in range(max_length):
#             new_beams = []
            
#             # For every existing beam, try all expansions
#             for seq, seq_score in beams:
#                 next_token_logits = self.get_next_token_logits(seq)
                
#                 next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)                
#                 topk_log_probs, topk_indices = torch.topk(next_token_log_probs, self.num_beams)
                
#                 for log_prob, token_id in zip(topk_log_probs, topk_indices):
#                     # Create new sequence by appending this token
#                     expanded_seq = torch.cat([seq, token_id.view(1,1)], dim=1)
                    
#                     # Update cumulative log probability
#                     new_score = seq_score + log_prob.item()
                    
#                     new_beams.append((expanded_seq, new_score))
            
#             # Out of the expanded candidate set, keep only top `num_beams` beams
#             new_beams.sort(key=lambda x: x[1], reverse=True)
#             beams = new_beams[: self.num_beams]
        
#         # The best beam will be the first after sorting by score (descending)
#         best_seq, best_score = beams[0]
        
#         # Decode and return
#         result = self.decode(best_seq[0])  # best_seq is shape [1, seq_len]
#         print("[BeamSearchManual] Final:", result)
#         return result


class BeamSearchSamplerManual(Sampler):
    def __init__(self, model_name: str = "gpt2-medium", num_beams: int = 3, early_stopping: bool = True):
        super().__init__(model_name=model_name)
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.eos_token_id = self.tokenizer.eos_token_id  # End-of-sequence token ID

    def __call__(self, prompt: str, max_length: int = 10):
        """
        Manual beam search decoding with early stopping:
         1. Start with a single beam containing the prompt (input_ids).
         2. Expand each beam, selecting top candidates.
         3. Stop if enough beams reach the EOS token.
        """
        input_ids = self.encode(prompt)  # shape: [1, seq_len]
        
        beams = [(input_ids, 0.0, False)]  # (token_ids, cumulative_log_prob, finished)

        for step in range(max_length):
            new_beams = []
            finished_beams = []

            for seq, seq_score, is_finished in beams:
                if is_finished:  
                    finished_beams.append((seq, seq_score))
                    continue  # Don't expand finished sequences

                next_token_logits = self.get_next_token_logits(seq)
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(next_token_log_probs, self.num_beams)

                for log_prob, token_id in zip(topk_log_probs, topk_indices):
                    expanded_seq = torch.cat([seq, token_id.view(1,1)], dim=1)
                    new_score = seq_score + log_prob.item()
                    
                    if token_id.item() == self.eos_token_id:  # If EOS is generated, mark as finished
                        finished_beams.append((expanded_seq, new_score))
                    else:
                        new_beams.append((expanded_seq, new_score, False))

            # Merge ongoing and finished beams, then keep top `num_beams`
            all_beams = finished_beams + new_beams
            all_beams.sort(key=lambda x: x[1], reverse=True)
            beams = all_beams[: self.num_beams]

            # **Early stopping condition**: if `early_stopping` is enabled and at least one beam is finished
            if self.early_stopping and any(b[0].squeeze()[-1].item() == self.eos_token_id for b in beams):
                break

        # Pick the best finished sequence if available, otherwise take the top beam
        best_seq, best_score, _ = max(beams, key=lambda x: x[1])

        result = self.decode(best_seq[0])  
        print("[BeamSearchManual] Final:", result)
        return result


# class BeamSearchSamplerManual(Sampler):
#     def __init__(self, model_name: str = "gpt2-medium", num_beams: int = 3, 
#                  early_stopping: bool = True, no_repeat_ngram_size: int = 2):
#         super().__init__(model_name=model_name)
#         self.num_beams = num_beams
#         self.early_stopping = early_stopping
#         self.no_repeat_ngram_size = no_repeat_ngram_size
#         self.eos_token_id = self.tokenizer.eos_token_id  # End-of-sequence token ID

#     def _block_repeated_ngrams(self, seq, next_token_logits):
#         """
#         Blocks tokens that would form a repeating n-gram.
#         """
#         if self.no_repeat_ngram_size <= 0 or seq.shape[1] < self.no_repeat_ngram_size:
#             return next_token_logits  # No need to apply if sequence is too short

#         seq_list = seq[0].tolist()  # Convert tensor to list

#         # Get last (no_repeat_ngram_size - 1) tokens to form potential n-grams
#         prefix_ngram = tuple(seq_list[-(self.no_repeat_ngram_size - 1):])

#         # If the sequence is too short, don't block anything
#         if len(prefix_ngram) < self.no_repeat_ngram_size - 1:
#             return next_token_logits

#         # Collect all previous n-grams of size `no_repeat_ngram_size`
#         ngram_counts = {}
#         for i in range(len(seq_list) - self.no_repeat_ngram_size + 1):
#             ngram = tuple(seq_list[i : i + self.no_repeat_ngram_size])
#             ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

#         # Block tokens that would create a repeating n-gram
#         for token_id in range(next_token_logits.shape[0]):
#             new_ngram = prefix_ngram + (token_id,)
#             if new_ngram in ngram_counts:
#                 next_token_logits[token_id] = -float("inf")  # Mask out the token

#         return next_token_logits

#     def __call__(self, prompt: str, max_length: int = 10):
#         """
#         Beam search decoding with early stopping and no-repeat n-grams.
#         """
#         input_ids = self.encode(prompt)  # shape: [1, seq_len]
        
#         beams = [(input_ids, 0.0, False)]  # (token_ids, cumulative_log_prob, finished)
#         finished_beams = []

#         for step in range(max_length):
#             new_beams = []

#             for seq, seq_score, is_finished in beams:
#                 if is_finished:
#                     finished_beams.append((seq, seq_score, True))
#                     continue  # Don't expand finished sequences

#                 next_token_logits = self.get_next_token_logits(seq)

#                 # Apply no-repeat n-gram blocking
#                 next_token_logits = self._block_repeated_ngrams(seq, next_token_logits)

#                 next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
#                 topk_log_probs, topk_indices = torch.topk(next_token_log_probs, self.num_beams)

#                 for log_prob, token_id in zip(topk_log_probs, topk_indices):
#                     expanded_seq = torch.cat([seq, token_id.view(1, 1)], dim=1)
#                     new_score = seq_score + log_prob.item()
                    
#                     if token_id.item() == self.eos_token_id:  # If EOS is generated, mark as finished
#                         finished_beams.append((expanded_seq, new_score, True))
#                     else:
#                         new_beams.append((expanded_seq, new_score, False))

#             # Merge ongoing and finished beams, then keep top `num_beams`
#             all_beams = finished_beams + new_beams
#             all_beams.sort(key=lambda x: x[1], reverse=True)
#             beams = all_beams[: self.num_beams]

#             # **Early stopping condition**: Stop if enough beams reach `<eos>`
#             num_finished = sum(1 for b in beams if b[2])
#             if self.early_stopping and num_finished >= 1:
#                 break

#         # Pick the best finished sequence if available, otherwise take the top beam
#         best_seq, best_score, _ = max(beams, key=lambda x: x[1])

#         result = self.decode(best_seq[0])  
#         print("[BeamSearchManual] Final:", result)
#         return result


if __name__ == "__main__":
    
    # BATCHING AND ADDING ATTENTION MASK DOES NOT WORK FOR GPT-2. UNLESS THE LENGTH OF THE TOKEN IS THE SAME
    prompt = [
        "Hello my name is",
        "John von Neumann",
    ]
    max_length = 64
    
    print("=== Greedy ===")
    greedy = GreedySampler()
    greedy(prompt, max_length=max_length)
    
    # print("=== TopK ===")
    # top_k = TopKSampler()
    # top_k(prompt, max_length)

    # print("\n=== Beam Search in Model ===")
    # beam_search = BeamSearchSampler(num_beams=3)
    # beam_search(prompt, max_length=max_length)
    
    
    # print("\n=== Beam Search Manual ===")
    # beam_search = BeamSearchSamplerManual(num_beams=3)
    # beam_search(prompt, max_length=max_length)
    
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