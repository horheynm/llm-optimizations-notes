import torch
import torch.nn as nn
from layer_norm import LayerNorm
from multi_head_attn import MultiHeadAttention
from feed_foward import FeedForwardBlock
from residual_connection import ResidualConnection


class DecoderBlock(nn.Module):
    """
    A single decoder block in a Transformer model.

    Parameters:
    - features (int): The embedding size (d_model).
    - self_attention (MultiHeadAttention): Self-attention module for target sequence.
    - cross_attention (MultiHeadAttention): Attention module for attending to encoder outputs.
    - feed_forward (FeedForwardBlock): Position-wise feedforward network.
    - dropout (float): Dropout rate.

    Forward Pass:
    - x: Target sequence embeddings. Shape (batch_size, tgt_seq_len, d_model).
    - encoder_output: Encoder's output features. Shape (batch_size, src_seq_len, d_model).
    - src_mask: Mask for the source input. Shape (batch_size, 1, 1, src_seq_len).
    - tgt_mask: Mask for the target input. Shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

    Returns:
    - x (Tensor): Processed target sequence embeddings.
    """

    def __init__(
        self,
        features: int,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        # Three residual connections for self-attention, cross-attention, and feed-forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder block.
        """

        # Self-attention with residual connection
        
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )

        # Cross-attention with encoder output and residual connection
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)
        )

        # Position-wise feedforward with residual connection
        x = self.residual_connections[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder composed of multiple DecoderBlocks.

    Parameters:
    - features (int): The embedding size (d_model).
    - layers (nn.ModuleList): A list of DecoderBlocks.

    Forward Pass:
    - x: Target sequence embeddings. Shape (batch_size, tgt_seq_len, d_model).
    - encoder_output: Encoder's output features. Shape (batch_size, src_seq_len, d_model).
    - src_mask: Mask for the source input. Shape (batch_size, 1, 1, src_seq_len).
    - tgt_mask: Mask for the target input. Shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

    Returns:
    - Normalized output tensor (batch_size, tgt_seq_len, d_model).
    """

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder, processing input through multiple decoder layers.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


if __name__ == "__main__":
    # Example configuration
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    d_ff = 2048  # Feed-forward hidden layer size
    num_layers = 6
    dropout = 0.1

    # Create dummy input tensors
    x = torch.randn(batch_size, seq_len, d_model)  # Target sequence embeddings
    encoder_output = torch.randn(batch_size, seq_len, d_model)  # Encoder output
    src_mask = torch.ones(batch_size, 1, seq_len)  # Source mask (no masking)
    tgt_mask = torch.ones(batch_size, seq_len, seq_len)  # Target mask (no masking)

    # Define layers
    self_attention = MultiHeadAttention(d_model, num_heads, dropout)
    cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
    feed_forward = FeedForwardBlock(d_model, d_ff, dropout)

    # Construct decoder blocks
    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model, self_attention, cross_attention, feed_forward, dropout)
        for _ in range(num_layers)
    ])

    # Initialize the decoder
    decoder = Decoder(d_model, decoder_blocks)

    # Forward pass
    output = decoder(x, encoder_output, src_mask, tgt_mask)

    # Print shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Ensure output shape matches input
    assert output.shape == x.shape, "Output shape does not match input shape!"
