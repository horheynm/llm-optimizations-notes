import torch
import torch.nn as nn
from layer_norm import LayerNorm
from multi_head_attn import MultiHeadAttention
from feed_foward import FeedForwardBlock
from residual_connection import ResidualConnection

class EncoderBlock(nn.Module):
    """
    A single transformer encoder block, consisting of:
    1. Multi-Head Self-Attention with a residual connection.
    2. Feed-Forward Network with a residual connection.

    Parameters:
    - features (int): The input/output feature dimension.
    - self_attention_block (MultiHeadAttention): Multi-head self-attention module.
    - feed_forward_block (FeedForwardBlock): Position-wise feed-forward network.
    - dropout (float): Dropout probability.

    Shape:
    - Input: (x, src_mask) -> [batch_size, seq_len, features]
    - Output: (x) -> [batch_size, seq_len, features]
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout),  # For Self-Attention
            ResidualConnection(features, dropout)   # For FeedForward Block
        ])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder block.

        - Applies self-attention with a residual connection.
        - Applies a feed-forward network with a residual connection.

        Args:
        - x (Tensor): Input tensor of shape [batch_size, seq_len, features]
        - src_mask (Tensor): Source mask tensor for attention.

        Returns:
        - Tensor: Output tensor of shape [batch_size, seq_len, features]
        """
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # Self-Attention
        x = self.residual_connections[1](x, self.feed_forward_block)  # Feed-Forward
        return x


class Encoder(nn.Module):
    """
    The Transformer Encoder consisting of multiple EncoderBlocks.

    Parameters:
    - features (int): The input/output feature dimension.
    - layers (nn.ModuleList): A list of EncoderBlock layers.

    Shape:
    - Input: (x, mask) -> [batch_size, seq_len, features]
    - Output: (x) -> [batch_size, seq_len, features]
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        - Passes input through each encoder block sequentially.
        - Applies final layer normalization.

        Args:
        - x (Tensor): Input tensor of shape [batch_size, seq_len, features]
        - mask (Tensor): Mask tensor for attention.

        Returns:
        - Tensor: Output tensor of shape [batch_size, seq_len, features]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Example Usage
if __name__ == "__main__":
    # Define model hyperparameters
    batch_size = 2
    seq_len = 5
    features = 512
    num_layers = 6
    d_ff = 2048
    num_heads = 8
    dropout = 0.1

    # Create sample input (random tensor)
    x = torch.randn(batch_size, seq_len, features)

    # Create a dummy attention mask (no masking)
    mask = torch.ones(batch_size, seq_len, seq_len)

    # Instantiate multi-head attention and feed-forward block
    self_attention_block = MultiHeadAttention(features, num_heads)
    feed_forward_block = FeedForwardBlock(features, d_ff, dropout)

    # Create multiple encoder blocks
    encoder_layers = nn.ModuleList([
        EncoderBlock(features, self_attention_block, feed_forward_block, dropout) for _ in range(num_layers)
    ])

    # Instantiate the Encoder
    encoder = Encoder(features, encoder_layers)

    # Forward pass through the encoder
    output = encoder(x, mask)

    # Print output shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Ensure input-output shape consistency
    assert output.shape == (batch_size, seq_len, features)
