"""

self-attention allows the model to relate words to each other

Query (Q) – Represents what the current token is "asking" for.
Key (K) – Represents how relevant other tokens are to this query.
Value (V) – Represents the actual information contained in the tokens. 

Q @ K -> similarity / attention_scores. measure of how much each token (query) should focus on every other token (key).

sofmax(QK)V -> probability * V -> The softmax scores act as attention weights, determining how much to focus on each token when aggregating information.

attention = softmax( QK / sqrt(d_k)) * V
d_k = d_model if one head


############# Example ##########################

sequence = "my name is" -> tokenize -> and assume has 3 input_ids, so at this iteration, the seq_length is 3

Q := [seq_len, d_model] = [3, 512] 

Then Q @ K is represented as -> [3, 512] @ [512, 3] -> [3, 3]
 
        my      name    is
my      v11     v12    v13  
name    v21     v22    v23
is      v31     v32    v33 


and then * 1/sqrt(d_k) and take the softmax of that -> softmax( QK / sqrt(d_k)) 

        my      name    is
my      p11     p12    p13  
name    p21     p22    p23
is      p31     p32    p33 

where sum(p) along the row sums up to 1. (p11 + p12 + p13 = 1)


V := [seq_len, 512] = [3, 512]

so 

attention = softmax( QK / sqrt(d_k)) * V -> [seq_len, d_model]
so each row after the matrix multiplication contains the meaning (embedding), position (positinal transformation) and the interaction with other words 


The attentions core Aij != Aji. How much token i (query) attends to token j (key) is not the same as the opposite. Attention tokens attends differntly so the matrix is not symmetric.


Attetnion is permutation invariant -> permutate the words, same score, differnt order, sum of the row = 1.
Attetion attending to itself Aij where i = j has the highest score -- its the most similar. 
"""


import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        
        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.Q_K = torch.nn.Linear(d_model, d_model)
        self.Q_V = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x := [batch, seq_len, d_model]
        
        Q = self.W_Q(x) # [batch, seq_len, d_model]
        K = self.Q_K(x)
        V = self.Q_V(x)
        
        attn_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / 
            torch.sqrt(torch.tensor(self.d_model, dtype=V.dtype))
        ) # [batch, seq_len, seq_len]
        
        attention_weights = torch.softmax(attn_scores, dim=-1)  # [bs, seq_len, seq_len]
        output = torch.matmul(attention_weights, V)  # [bs, seq_len, d_model]
        
        return output, attn_scores

        
batch_size = 2
seq_len = 3
d_model = 512

x = torch.rand(batch_size, seq_len, d_model)

self_attention = SelfAttention(d_model)

out, scores = self_attention(x)

print(
    out.shape,  # [bs, seq_len, d_model]
    scores.shape # [bs, seq_len, seq_len]
)



############### Attention is permutation invariant ###############

import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    d_k = Q.shape[-1]  # Key dimension
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_probs = F.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_probs, V)
    return output, attention_probs

# Create a toy input sequence (3 tokens, each with 4 features)
torch.manual_seed(42)
X = torch.randn(3, 4)  # 3 tokens, 4-dimensional embeddings

# Compute Q, K, V using simple linear projections
W_Q = torch.randn(4, 4)
W_K = torch.randn(4, 4)
W_V = torch.randn(4, 4)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Compute self-attention output and attention scores
original_output, original_attention = self_attention(Q, K, V)

# Shuffle the input sequence
idx = torch.randperm(3)  # Random permutation of indices
X_shuffled = X[idx]
print(idx)

# Compute Q, K, V for shuffled input
Q_shuffled = X_shuffled @ W_Q
K_shuffled = X_shuffled @ W_K
V_shuffled = X_shuffled @ W_V

# Compute self-attention again
shuffled_output, shuffled_attention = self_attention(Q_shuffled, K_shuffled, V_shuffled)

# Check if the attention scores remain the same (permutation invariant)
print("Original Attention Scores:\n", original_attention)
print("Shuffled Attention Scores:\n", shuffled_attention)
print("\nAre the attention scores identical?", torch.allclose(original_attention, shuffled_attention, atol=1e-6))


