import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Normalization is applied across all features in a single instance, 
    meaning statistics are computed across the channel (C) and spatial 
    dimensions (H, W) for each sample individually.
    
    This implementation normalizes over the last dimension of the input
    by default, and applies learnable scaling (alpha) and shifting (beta).
    
    
    Transformers use LayerNorm to stabilize training and improve convergence. The key benefits are:

    Reduces Internal Covariate Shift: Helps prevent large variations in activations that can destabilize training.
    Improves Gradient Flow: By normalizing inputs, it prevents vanishing or exploding gradients.
    Works Well with Self-Attention: Since self-attention computes weighted sums, normalizing features ensures stability.
    Batch Size Independence: Unlike BatchNorm, LayerNorm does not rely on batch statistics, making it ideal for small batches or variable-length sequences.
    
    Layer Norm:
        Q[i,j,:] -> normalize the feature-dimension
    
    Batch norm
        Q[:,:, k] -> Normalize the seq, token dim
        
    Do not use BN bc:
    1. Sequence Length Variability
        Transformers process sequences of different lengths (e.g., sentences, paragraphs). BatchNorm computes statistics (mean and variance) across the batch, but sequences can have different lengths, making it difficult to apply consistent normalization.

        In CNNs, images have fixed sizes, so BatchNorm can compute batch-wise statistics reliably.
        In NLP tasks, each input may have a different number of tokens, making batch statistics unreliable.
        📌 Issue: BatchNorm’s batch-level statistics are inconsistent for variable-length sequences.


    2. Small and Dynamic Batch Sizes
        Transformers are often trained with:

        Small batch sizes (due to high memory requirements in large models).
        Dynamic batch sizes (e.g., varying numbers of tokens per batch in NLP).
        BatchNorm relies on large batch statistics (mean & variance) for stable normalization. However, with small or varying batch sizes:

        Batch statistics become noisy, leading to training instability.
        The model becomes more sensitive to small batch variations.
        
    3. Non-Independent Token Processing in Transformers
        Transformers process tokens independently within a batch, meaning different tokens attend to different parts of the sequence.
            However, BatchNorm computes normalization across the batch dimension, introducing unintended dependencies between tokens.

        Self-attention relies on token-to-token interactions.
        BatchNorm forces dependencies between tokens across different sequences, which is undesirable.
        📌 Issue: BatchNorm introduces inter-sequence dependencies, which interfere with self-attention.

    """
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [batch_size, ..., feature_dim]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return self.alpha * (x - mean) / (std + self.epsilon) + self.beta


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 3
    feature_dim = 4
    
    # Create a random tensor: shape [batch_size, seq_len, feature_dim]
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # Instantiate the custom LayerNorm
    layer_norm = LayerNorm()

    # Forward pass
    output = layer_norm(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    # The shape should match the input shape
    assert x.shape == output.shape
    
    print(x)
    print(output)
