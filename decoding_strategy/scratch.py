
import torch
import AutoModelForCausalLM, AutoTokenizer



class Sampler:
    def __init__(self, model_name: str= "gpt2-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def encode(self, text: str) -> torch.Tensor:
        tensor = self.tokenizer.encode(
            text,
            return_tensors="pt",
        )
        return tensor
    
    def decode(self, input_ids: torch.Tensor) -> str:
        decoded_text = self.tokenizer.decode(
            input_ids
        )
        return decoded_text
    
    def get_next_token_prob(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Given encoded tensor, run forward pass and get the logits of the newly generated
        """
        
        with torch.no_grad():
            out = self.model(
                input_ids
            )
        logits: torch.Tensor = out.logits # [batch, seq_len, vocab_size]
        
        # get the latest
        return logits[0, -1, :]
    

class GreedySampler(Sampler):
    def __call__(self, prompt: str, max_length: int = 10):
        
        input_ids = self.encode(prompt)
        
        for _ in range(max_length):
            
            latest_logits = self.get_next_token_prob(input_ids)
            
            max_input_id = torch.argmax(latest_logits, dim=-1).item()
            input_ids += max_input_id
            
        
        print(
            self.decode(input_ids)
        )
            
            
        
        
        
        