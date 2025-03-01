import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention as described in the "Attention Is All You Need" paper.
    
    Parameters:
    - d_model (int): Embedding dimension of the input.
    - num_heads (int): Number of attention heads.
    - dropout (float): Dropout rate for attention probabilities.

    Shape:
    - Input: (batch_size, seq_len, d_model)
    - Output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for queries, keys, values, and output
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
        """
        Computes scaled dot-product attention.
        
        Args:
        - query (Tensor): (batch, num_heads, seq_len, d_k)
        - key (Tensor): (batch, num_heads, seq_len, d_k)
        - value (Tensor): (batch, num_heads, seq_len, d_k)
        - mask (Tensor, optional): Mask tensor for padding or causal masking.
        - dropout (nn.Dropout, optional): Dropout layer to apply on attention scores.

        Returns:
        - output (Tensor): Attention-weighted values.
        - attention_weights (Tensor): Softmax attention scores for visualization.
        """
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # q.shape= torch.Size([2, 8, 5, 64])
        attention_weights = torch.softmax(attention_scores, dim=-1) 

        if dropout is not None:
            attention_weights = dropout(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
        - q (Tensor): Query tensor of shape (batch, seq_len, d_model)
        - k (Tensor): Key tensor of shape (batch, seq_len, d_model)
        - v (Tensor): Value tensor of shape (batch, seq_len, d_model)
        - mask (Tensor, optional): Mask tensor to prevent attending to certain positions.

        Returns:
        - output (Tensor): (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = q.shape

        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape for multi-head processing
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # [b,s,d_model, d_model // num_head]
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        breakpoint()
        # Compute attention
        x, self.attention_scores = self.scaled_dot_product_attention(query, key, value, mask, self.dropout) # torch.Size([2, 8, 5, 5])

        # Reshape and concatenate heads
        
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply final linear transformation
        return self.w_o(x)


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, d_model, num_heads = 2, 5, 512, 8

    x = torch.randn(batch_size, seq_len, d_model)  # Random input tensor
    mask = None  # Example mask (optional)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    output = mha(x, x, x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert x.shape == output.shape  # Ensures input-output shape consistency

"""
# **What Does Multi-Head Attention Do?**
Multi-Head Attention (MHA) is a core mechanism in **Transformers** that allows the model to **attend to different parts of the input sequence simultaneously**. It enhances the model's ability to understand **context and relationships** between tokens (words) by using **multiple attention heads**.

## **How Does Multi-Head Attention Work?**
Multi-Head Attention extends **self-attention** by running multiple attention mechanisms in parallel. Here’s how it works:

1. **Linear Projections**  
   - The input embeddings (each token) are **transformed** into three different vectors:
     - **Query (Q)**
     - **Key (K)**
     - **Value (V)**
   - This transformation happens separately for each **attention head**.

2. **Scaled Dot-Product Attention (For Each Head)**  
   - The attention score between tokens is computed as:

     \[
     \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
     \]

   - Scaling by \( \sqrt{d_k} \) ensures **stable gradients**.

3. **Parallel Attention Heads**  
   - Instead of a **single** attention mechanism, multiple heads attend to **different aspects** of the sequence.
   - Each head learns a **different representation**, capturing **different semantic relationships**.

4. **Concatenation and Final Projection**  
   - The outputs from all heads are **concatenated** and **projected back** to the original embedding size.

---

## **What Is the Purpose of Multi-Head Attention?**
Multi-Head Attention **improves Transformers in several ways**:

### **1️⃣ Allows the Model to Attend to Multiple Contexts Simultaneously**
- Each head **focuses on different parts of the input**.
- Example: In a translation model, one head might focus on **the subject**, another on **the verb**, and another on **tense information**.

📌 **Without MHA:** The model would only focus on **one** relationship at a time, limiting its expressiveness.

---

### **2️⃣ Improves Information Flow in Deep Models**
- Using multiple attention heads **reduces information bottlenecks**.
- Helps the model **propagate** different features **across multiple layers**.

📌 **Without MHA:** The model may fail to learn **complex relationships**, making it less powerful.

---

### **3️⃣ Increases Expressiveness and Robustness**
- Since different heads **learn different patterns**, the model can capture a **richer representation** of the data.
- Reduces over-reliance on any **single pattern**.

📌 **Without MHA:** The model might struggle with **long dependencies**, leading to **worse accuracy** in tasks like translation and summarization.

---

### **4️⃣ Reduces Overfitting**
- Splitting the attention mechanism across multiple heads **acts as an implicit regularization technique**.
- Prevents the model from **overfitting** to specific token relationships.

📌 **Without MHA:** The model might overfit to **one type of pattern**, generalizing poorly to new data.

---

## **What Happens in Transformers Without Multi-Head Attention?**
If we **remove Multi-Head Attention**, the Transformer loses many of its benefits:

🚫 **Single-Head Attention Would Be Limited**  
- The model would only **attend to one aspect** of the input.
- The representation would be **less expressive**.

🚫 **Loss of Long-Range Dependencies**  
- Multi-Head Attention allows Transformers to model **relationships between words far apart**.
- Without it, the model would struggle to understand **context in long sequences**.

🚫 **Reduced Performance in NLP Tasks**  
- Tasks like **machine translation, summarization, and question-answering** would suffer.
- Transformers would **perform more like simple RNNs**, which struggle with long-term dependencies.

🚫 **No Multi-Perspective Learning**  
- Different heads capture **different relationships**.
- Without MHA, the model would **miss subtle relationships** between words.

---

## **Conclusion**
✅ **Multi-Head Attention is crucial in Transformers because it:**  
✔️ **Enables multiple perspectives on the input**  
✔️ **Enhances expressiveness by capturing different relationships**  
✔️ **Improves learning of long-range dependencies**  
✔️ **Prevents overfitting and information bottlenecks**  

🔥 **Without it, Transformers would lose their power**, making them **less effective for NLP, vision, and other tasks**. 🚀


"""