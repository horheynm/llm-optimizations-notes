
"""
https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png
"""

import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    Section 3.4: Embeddings and Softmax of the Attention Is All You Need paper:
    https://arxiv.org/pdf/1706.03762
    
    scaling of sqrt(d_model) -> stablize gradients. Without, then embeddings may have a lower mag that can slow down
        learning. When initializing the emb weights, the variance is  ~1/d_model, if sqrt(d_model) / d_model, the variance is 
        roughly 1
        
    This is done because Transformer models use residual connections and layer normalization, which assume inputs have a variance around 1. Without the
        sqrt(d_model) factor, embedding values would be too small, leading to slower optimization.
    
    
    Math:
    - elements are sampled from x~N(0, signa^2) using Glorot init, which has the property

        sigma^2 = 1 / d_model
        
    This keeps the variance small to prevent exploding activations in NN
    
    - variance
        Var[c * x] = c^2 Var[x], with c = scaling factor = sqrt(d_model) = 1
        
    without the scaling, then as d_model increase, the variance decrease exp -> vanishing activations. 
    
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model: dim of the model
        vocab_size: total number of vocabs the layer can learn
        """
        super().__init__()
        
        self.d_model  = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        """
        x.shape                 = [batch, seq_len,]
        self.forward(x).shape   = [batch, seq_len, d_model]
        """
        
        return self.embedding(x) * torch.tensor(self.d_model ** 0.5)
        


def run():
    batch_size = 2    
    seq_len = 5        
    vocab_size = 10_000  
    d_model = 512  

    x = torch.randint(0, vocab_size, (batch_size, seq_len)) # 0 to vocab_size, shape of [batch, seq]
    
    embedding_layer = InputEmbedding(d_model, vocab_size)
    output = embedding_layer(x)

    print("\nOutput shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)
    print("Output embeddings (sample):", output[0, 0, :5])  # Show first 5 values of first token embedding