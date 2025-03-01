


import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    """
    Section 3.5 of the Attention Is All You Need paper:
    https://arxiv.org/pdf/1706.03762
    
    """
    
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        position_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1) # [max_len, 1]     
        
        # 10000^(2i/d_model)
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))

        position_encoding[:, 0::2] = torch.sin(position * denom) # 0, 2, 4
        position_encoding[:, 1::2] = torch.cos(position * denom) # 1, 3, 5

        # add batch dim
        position_encoding = position_encoding.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", position_encoding)
        
    
    def forward(self, x):
        max_len = x.size(1)

        return self.dropout(x + self.pe[:, :max_len])
    

if __name__ == "__main__":
    d_model = 512  # embedding dimension
    max_len = 10   
    batch_size = 2 
    dropout = 0.1  


    x = torch.randn(batch_size, max_len, d_model)

    pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
    x_encoded = pos_encoding(x)

    print("Input shape:", x.shape)
    print("Output shape:", x_encoded.shape) 
    assert x.shape == x_encoded.shape