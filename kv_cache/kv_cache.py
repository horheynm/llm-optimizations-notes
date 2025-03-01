"""


MEMORY:
    len([key, value]) * 16 bits / 8 bits/byte * len(layers) * len(attn_head) * dim(attn_head)
    = 2 * 2 * n_layers * n_head * d_head




"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


##############################
# KV Cache Helper
##############################
class KVCache:
    """
    A simple key/value cache for storing past keys and values for each layer.
    This cache stores entries per layer (using layer index as key).
    """

    def __init__(self):
        self.cache = {}

    def get(self, layer_idx):
        return self.cache.get(layer_idx, None)

    def update(self, layer_idx, k, v):
        # If a cache already exists, concatenate the new k/v along the sequence dimension.
        if layer_idx in self.cache:
            cached_k, cached_v = self.cache[layer_idx]
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        # Detach to avoid backpropagating into the cached values.
        self.cache[layer_idx] = (k.detach(), v.detach())

    def clear(self):
        self.cache = {}


##############################
# Multi-Head Self-Attention with KV Cache
##############################
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        """
        hidden_size: model dimension
        num_heads: number of attention heads (assumes hidden_size is divisible by num_heads)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Combined Q, K, V projection.
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer_idx=None, kv_cache: KVCache = None):
        """
        x: (batch, seq_len, hidden_size)
        layer_idx: identifier for this layer (used for caching)
        kv_cache: instance of KVCache to retrieve and update cached keys/values.
        """
        batch, seq_len, _ = x.size()

        # Compute combined QKV and split.
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*hidden_size)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each is (batch, seq_len, hidden_size)

        # Reshape for multi-head attention.
        # New shape: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # If a KV cache is provided, retrieve previous keys/values and update.
        if kv_cache is not None and layer_idx is not None:
            cached = kv_cache.get(layer_idx)
            if cached is not None:
                cached_k, cached_v = cached
                # Concatenate along the sequence length dimension (dim=2).
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            # Update the cache with the concatenated keys and values.
            kv_cache.update(layer_idx, k, v)

        # Scaled dot-product attention.
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        )  # (batch, num_heads, seq_len, seq_len_total)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output.
        attn_output = torch.matmul(
            attn_weights, v
        )  # (batch, num_heads, seq_len, head_dim)
        # Reshape: combine heads.
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.hidden_size)
        )
        output = self.out_proj(attn_output)
        return output


##############################
# Decoder Layer (Transformer Block)
##############################
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer_idx=None, kv_cache: KVCache = None):
        # Self-attention sub-layer with residual connection.
        attn_out = self.attn(self.ln1(x), layer_idx=layer_idx, kv_cache=kv_cache)
        x = x + self.dropout(attn_out)
        # Feed-forward sub-layer with residual connection.
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)
        return x


##############################
# Decoder (Stack of Transformer Blocks)
##############################
class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, x, kv_cache: KVCache = None):
        # Pass through each decoder layer.
        for i, layer in enumerate(self.layers):
            x = layer(x, layer_idx=i, kv_cache=kv_cache)
        x = self.final_ln(x)
        return x


##############################
# Token Embedding + Positional Encoding (Encoder)
##############################
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # Learnable positional embeddings.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        """
        x = self.token_embedding(input_ids)  # (batch, seq_len, hidden_size)
        seq_len = x.size(1)
        # Add positional embeddings.
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        return x


##############################
# LLaMA Model
##############################
class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        max_seq_len,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        """
        A simplified LLaMA-style model.
        """
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_size, max_seq_len, dropout)
        self.decoder = Decoder(num_layers, hidden_size, num_heads, mlp_ratio, dropout)
        # Language modeling head (tied to embeddings in real implementations).
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, kv_cache: KVCache = None):
        """
        input_ids: (batch, seq_len)
        kv_cache: instance of KVCache (for incremental generation)
        """
        x = self.encoder(input_ids)
        x = self.decoder(x, kv_cache=kv_cache)
        logits = self.lm_head(x)
        return logits


##############################
# Weight Loader Helper
##############################
def load_llama_weights(model: nn.Module, checkpoint_path: str):
    """
    Loads the checkpoint weights into the model.

    Args:
        model (nn.Module): The LLaMA model instance.
        checkpoint_path (str): Path to the checkpoint file (e.g., a .pt or .pth file).

    This function assumes that the checkpoint's state dict either directly matches
    the model's state dict, or contains a sub-dictionary (e.g., under "model") that needs
    to be extracted. It also provides an example of how to remap keys if they contain
    prefixes (adjust as needed).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # If the checkpoint is wrapped in a dict with a "model" key, extract it.
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Example remapping: remove a "model." prefix from keys if present.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model.") :]
        new_state_dict[new_key] = value

    # Load the state dict into the model.
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded checkpoint from:", checkpoint_path)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    print("Model weights loaded successfully.")


##############################
# Text Generation Helper
##############################
@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    kv_cache: KVCache = None,
):
    """
    Generates text autoregressively from a given prompt using greedy decoding.

    Args:
        model (nn.Module): The LLaMA model.
        tokenizer: The tokenizer to encode/decode text.
        prompt (str): The input prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        kv_cache (KVCache): Optional key/value cache for incremental generation.

    Returns:
        str: The generated text.
    """
    model.eval()
    # Tokenize input prompt.
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(next(model.parameters()).device)
    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated, kv_cache=kv_cache)
        # Get logits for the last generated token.
        next_token_logits = logits[:, -1, :]
        # Greedy decoding: pick the token with the highest probability.
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        # If the EOS token is generated, stop early.
        if next_token.item() == tokenizer.eos_token_id:
            

    # Decode the generated tokens back to text.
    return tokenizer.decode(generated[0], skip_special_tokens=True)


##############################
# Example Usage
##############################
if __name__ == "__main__":
    # Example hyperparameters (illustrative for a LLaMA-7B-style model)
    vocab_size = 32000
    hidden_size = 4096  # Model dimension
    num_layers = 32  # Number of transformer layers
    num_heads = 32  # Number of attention heads
    max_seq_len = 2048
    dropout = 0.1

    # Create the model.
    model = LlamaModel(
        vocab_size, hidden_size, num_layers, num_heads, max_seq_len, dropout=dropout
    )

    checkpoint_path = "/home/gohashi/.llama/checkpoints/Llama-2-7b/checklist.chk"
    try:
        load_llama_weights(model, checkpoint_path)
    except FileNotFoundError as e:
        print(e)

    # Initialize an empty KV cache for autoregressive generation.
    kv_cache = KVCache()

    # Load the tokenizer.
    # Make sure you have installed transformers (pip install transformers)
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        "/home/gohashi/.llama/checkpoints/Llama-2-7b/tokenizer.model"
    )

    # Prompt for user input.
    prompt = input("Enter a prompt: ")
    print("\nGenerating text...\n")
    generated_text = generate_text(
        model, tokenizer, prompt, max_new_tokens=50, kv_cache=kv_cache
    )
    print("Generated text:")
    print(generated_text)
