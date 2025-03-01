import torch
import torch.nn as nn
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from input_embedding import InputEmbedding
from positional_encoding import PositionalEncoding
from projection_layer import ProjectionLayer
from multi_head_attn import MultiHeadAttention
from feed_foward import FeedForwardBlock

class Transformer(nn.Module):
    """
    Implements a Transformer model consisting of an Encoder-Decoder architecture.

    Parameters:
    - encoder (Encoder): Transformer encoder module.
    - decoder (Decoder): Transformer decoder module.
    - src_embed (InputEmbedding): Input embedding layer for the source sequence.
    - tgt_embed (InputEmbedding): Input embedding layer for the target sequence.
    - src_pos (PositionalEncoding): Positional encoding for the source sequence.
    - tgt_pos (PositionalEncoding): Positional encoding for the target sequence.
    - projection_layer (ProjectionLayer): Final linear layer mapping decoder output to vocabulary size.

    Methods:
    - encode: Processes the source sequence through the embedding, positional encoding, and encoder.
    - decode: Processes the target sequence through the embedding, positional encoding, and decoder.
    - project: Maps the decoder output to vocabulary logits.

    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
        - src (Tensor): Source input tensor (batch_size, src_seq_len).
        - src_mask (Tensor): Source mask (batch_size, 1, 1, src_seq_len).

        Returns:
        - Tensor: Encoded source representation (batch_size, src_seq_len, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the target sequence.

        Args:
        - encoder_output (Tensor): Encoder output (batch_size, src_seq_len, d_model).
        - src_mask (Tensor): Mask for the source input (batch_size, 1, 1, src_seq_len).
        - tgt (Tensor): Target input tensor (batch_size, tgt_seq_len).
        - tgt_mask (Tensor): Target mask (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
        - Tensor: Decoder output (batch_size, tgt_seq_len, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps the decoder output to vocabulary logits.

        Args:
        - x (Tensor): Decoder output (batch_size, tgt_seq_len, d_model).

        Returns:
        - Tensor: Vocabulary logits (batch_size, tgt_seq_len, vocab_size).
        """
        return self.projection_layer(x)



def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


if __name__ == "__main__":

    src_vocab_size = tgt_vocab_size = 30000
    src_seq_len = tgt_seq_len = 4096
    
    model  = build_transformer(src_vocab_size, tgt_vocab_size,  src_seq_len, tgt_seq_len)
    print(model)
    
