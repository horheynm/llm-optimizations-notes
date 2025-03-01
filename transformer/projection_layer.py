import torch
import torch.nn as nn

"""
The final linear layer in transformers
"""
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass: Projects the transformer hidden states to vocabulary logits.
        
        Args:
        - x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
        - Tensor: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        return self.proj(x)

if __name__ == "main":
    # Example Usage
    batch_size, seq_len, d_model, vocab_size = 2, 5, 512, 10000
    x = torch.randn(batch_size, seq_len, d_model)

    projection_layer = ProjectionLayer(d_model, vocab_size)
    logits = projection_layer(x)

    print("Input shape:", x.shape)  # [2, 5, 512]
    print("Output shape:", logits.shape)  # [2, 5, 10000]

"""
# **What is the `ProjectionLayer` and Where is it Used?**

The `ProjectionLayer` is a crucial component used in **transformer-based language models**, particularly at the **final stage** of text generation tasks like **machine translation, text completion, and speech-to-text systems**.
It **maps the hidden representations of tokens into vocabulary logits**, allowing the model to predict the next token in a sequence.

---

## **📌 What Does `ProjectionLayer` Do?**

In transformer models, each token is represented as a high-dimensional vector after passing through multiple layers of self-attention and feedforward transformations. 
However, this hidden representation **does not directly correspond to words or tokens**. The `ProjectionLayer` acts as a bridge by converting these representations into **logits over the vocabulary**.

### **Breakdown of Components:**
- **`d_model`** → The dimensionality of each token's hidden representation (e.g., `512` in standard transformer models).
- **`vocab_size`** → The total number of unique tokens in the vocabulary (e.g., `50,000` for large NLP models like BERT and GPT).
- **`nn.Linear(d_model, vocab_size)`** → A fully connected layer that **projects each token’s hidden state into a probability distribution over all possible tokens**.

After this projection, **softmax is applied** to transform the logits into **probability scores**, and the token with the **highest probability is selected as the next word**.

---

## **📌 Where is `ProjectionLayer` Used?**

This layer plays an essential role in **transformer-based NLP models**, particularly in:

1. **Decoder Output Processing:**
   - In **sequence-to-sequence tasks** like **machine translation (e.g., English to French)**, the decoder processes input and generates **hidden states**.
   - These hidden states need to be **converted into a predicted word/token**—this is where `ProjectionLayer` is applied.

2. **Autoregressive Text Generation:**
   - Models like **GPT (Generative Pre-trained Transformer)** generate text token by token.
   - Each token's **hidden representation is projected onto the vocabulary space**, and the most likely next word is selected.

### **📌 How It Works in a Transformer Model**
1. The **decoder outputs hidden states** of shape `(batch_size, seq_len, d_model)`.
2. The `ProjectionLayer` **maps these hidden states to vocabulary logits** of shape `(batch_size, seq_len, vocab_size)`.
3. **Softmax is applied** to get a probability distribution.
4. The token with the **highest probability is selected as the next word**.



---

## **📌 What Happens If We Remove `ProjectionLayer`?**

❌ **The model won’t generate words** → The decoder only outputs **numerical embeddings**, which do not correspond to actual words.  
❌ **No mapping to vocabulary** → There would be no way to determine which word should be the next output.  
❌ **Loss cannot be computed** → In training, models compare predicted logits to actual tokens in the dataset. Without the `ProjectionLayer`, this step would be impossible.  

### **🚀 Conclusion: Why is `ProjectionLayer` Important?**
The `ProjectionLayer` is **essential for converting hidden representations into actual words**. Without it, **transformers wouldn’t be able to generate meaningful output**. It is the final piece that enables models like **GPT, BERT, and T5** to **produce human-readable text** from their deep internal computations. 🚀



"""