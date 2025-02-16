"""
In large language model inference, input tokens are very crucially processed efficiently to reduce latency. 
When a prompt is received, first the prefill phase where all input tokens are processed simultaneously, is conducted.
This step utilizes self-attention such that every token can interact with every other token once in parallel.
Given that all input tokens are provided beforehand, this phase is highly optimized with matrix multiplications and thus much quicker than the decoding phase, where input tokens must be generated one at a time.
For models like GPT-4 and LLaMA long prompt handling fast and efficient just too requires the prefill step optimization.

If prefill were handled sequentially—process one token at a time—it would lead to unnecessary latency, particularly with large inputs.
Instead, modern implementations compute queries, keys, and values QKV for all tokens in parallel which significantly speeds up inference.
This optimization helps LLMs process user inputs quickly and respond faster in applications like chatbots, AI assistants, and content generation applications.
Efficient prefill ensures that most of the computation happens upfront so that the model can spend more time on autoregressive decoding, which is resource-intensive in response generation.


Why Does Parallel Encoding Work?
* Self-attention allows each token to attend to all others simultaneously.
* Q, K, V matrices are computed in parallel, reducing iteration level calculations.
* Batch processing uses matrix multiplications, making it significantly faster.


The causal mask ensures that each token in a sequence can only attend to itself and the tokens that come before it, effectively blocking attention to future tokens. 
This is crucial in autoregressive or causal language modeling scenarios, where each newly generated token must not “peek” at tokens that haven’t been generated yet. 
By adding a -∞ penalty (or very large negative number) to the attention logits for any positions beyond the current token’s index, 
 the softmax turns those positions into zeros—meaning they can’t contribute to the output.


"""


import torch
import torch.nn as nn
import math

SEED = 42
torch.manual_seed(SEED)  

class BaseEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def _causal_mask(self, seq_len):
        """
        Prevent attention to see future tokens
        """
        # 1) Create a matrix of 0s and 1s where upper triangular (j>i) are 1
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # 2) Convert to float and fill masked positions with -inf
        mask = mask.float().masked_fill_(mask, float('-inf'))
        return mask


class EncodeSequential(BaseEncoder):
    """
    Causal attention implemented sequentially (i.e., token by token).
    Each token t can only see x[:, :t+1, :].
    """
    def __init__(self, d_model, seq_len):
        super().__init__(d_model)
        self.seq_len = seq_len

    def forward(self, x):
        """
        For each time-step t in [0..seq_len-1], 
        - Construct Q from x[:, t].
        - Construct K, V from x[:, :t+1].
        - Compute attention over t+1 positions.
        """
        batch_size, seq_len, d_model = x.shape
        
        outputs = torch.zeros(batch_size, seq_len, d_model, device=x.device)

        for t in range(seq_len):
            # Query: shape (batch_size, d_model)
            q = self.W_q(x[:, t, :])
            # Expand Q for batch matrix multiplication: (batch_size, 1, d_model)
            q = q.unsqueeze(1)

            # K, V: only up to t-th position => (batch_size, t+1, d_model)
            k = self.W_k(x[:, :t+1, :])
            v = self.W_v(x[:, :t+1, :])

            # attention_scores shape => (batch_size, 1, t+1)
            attention_scores = torch.bmm(
                q, k.transpose(1, 2)
            ) / math.sqrt(self.d_model)

            # Softmax over the last dimension => (t+1)
            attention_probs = torch.softmax(attention_scores, dim=-1)

            # attended_values => (batch_size, 1, d_model)
            attended_values = torch.bmm(attention_probs, v)

            # Remove singleton dimension => shape (batch_size, d_model)
            outputs[:, t, :] = attended_values.squeeze(1)

        return outputs


class EncodeParallel(BaseEncoder):
    """
    Causal attention in parallel using a lower-triangular mask.
    Each token sees only up to its own position within a single forward pass.
    """
    def __init__(self, d_model):
        super().__init__(d_model)

    def forward(self, x):
        """
        - Compute Q, K, V for the entire sequence.
        - Apply a causal mask so that token i cannot attend to tokens j > i.
        """
        batch_size, seq_len, d_model = x.shape

        q = self.W_q(x)  # (batch_size, seq_len, d_model)
        k = self.W_k(x)  # (batch_size, seq_len, d_model)
        v = self.W_v(x)  # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) x (batch_size, d_model, seq_len) -> (batch_size, seq_len, seq_len)
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_model)

        # Build a causal mask of shape (seq_len, seq_len) and broadcast to (batch_size, seq_len, seq_len).
        mask = self._causal_mask(seq_len).to(x.device)  # (seq_len, seq_len)
        
        # Expand for batch dimension if needed: (1, seq_len, seq_len) => broadcast to (batch_size, seq_len, seq_len)
        attention_scores = attention_scores + mask

        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Multiply by V => (batch_size, seq_len, d_model)
        attended_values = torch.bmm(attention_probs, v)

        return attended_values



if __name__ == "__main__":
    
    batch_size, seq_len, d_model = 2, 4, 8
    x = torch.randn(batch_size, seq_len, d_model)

    sequential_model = EncodeSequential(d_model, seq_len)
    parallel_model = EncodeParallel(d_model)

    # Ensure both have the same initialized weights
    parallel_model.load_state_dict(sequential_model.state_dict())

    with torch.no_grad():
        out_seq = sequential_model(x)
        out_par = parallel_model(x)

    print("Sequential output shape: ", out_seq.shape)
    print("Parallel output shape: ", out_par.shape)

    if torch.allclose(out_seq, out_par, atol=1e-6):
        print("Output matches with parallel and seqential, expected")
    else:
        print("Outputs do not match, not expected")
        diff = (out_seq - out_par).abs().max()
        print(f"Max difference: {diff}")
