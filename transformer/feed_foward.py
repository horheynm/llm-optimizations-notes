import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """
    Implements the Position-wise FeedForward network used in Transformer architectures.
    
    This consists of two linear layers with a ReLU activation in between and optional dropout.
    It is applied independently to each position (token) in the sequence.
    
    Parameters:
    - d_model (int): The input and output dimension of the model.
    - d_ff (int): The hidden dimension of the feedforward network (usually larger than d_model).
    - dropout (float): Dropout probability to prevent overfitting.
    
    Shape:
    - Input: [batch_size, seq_len, d_model]
    - Output: [batch_size, seq_len, d_model]
    

    Purpose of the FeedForward Layer:
        Enhances Representation Power

        Self-attention captures contextual relationships between tokens.
        The feedforward layer transforms individual token embeddings, allowing the model to learn complex features.
        Expands Dimensionality for Better Learning

        The first linear layer expands the dimension (e.g., from 512 → 2048 in BERT).
        The second linear layer compresses it back (2048 → 512).
        This creates a "bottleneck" that helps the model learn richer representations.
        Applies Non-Linearity (ReLU)

        The ReLU activation introduces non-linearity, making the model capable of learning more complex patterns.
        Stabilizes Training (Dropout)

        Dropout prevents overfitting, ensuring robustness.
        
        
    If we remove the FeedForward layer from a transformer:

        🚫 Limited Token Transformation → Self-attention captures context but cannot transform individual tokens independently.
        🐌 Reduced Model Expressiveness → The model cannot refine learned features.
        📉 Weaker Performance → Tasks like translation, text generation, and understanding suffer lower accuracy.
        Example:
        Without the FeedForward layer, the transformer only relies on self-attention, leading to weaker token representations.
        In BERT, GPT, and T5, removing FFN degrades performance significantly.
        
        
    4️⃣ Non-Linearity in the Transformer FeedForward Layer
        In the FeedForward layer of a Transformer, the two linear layers are separated by a non-linear activation function (ReLU):

        FFN(x) = W2 * RELU(W1*x + b1) + b2

        First linear layer W1 expands dimensions (e.g., 512 → 2048).
        ReLU introduces non-linearity, allowing the model to learn more complex patterns.
        Second linear layer W2 projects it back to the original dimension (2048 → 512).

        📌 Without ReLU, the two linear layers collapse into a single linear function, making the FeedForward layer useless! 😱

        5️⃣ What Happens If We Remove Non-Linearity?
        ❌ The model would become just a stack of linear layers.
        ❌ It would behave like a simple matrix transformation (like logistic regression).
        ❌ The transformer would lose its ability to model complex relationships.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # Activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feedforward network.
        
        Steps:
        1. Apply first linear transformation (d_model → d_ff)
        2. Apply ReLU activation
        3. Apply dropout (optional)
        4. Apply second linear transformation (d_ff → d_model)
        
        Returns:
        - Transformed tensor of the same shape as input.
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, d_model, d_ff = 2, 5, 512, 2048
    x = torch.randn(batch_size, seq_len, d_model)  # Random input

    feedforward = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=0.1)
    output = feedforward(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert x.shape == output.shape  # Ensure input-output shape remains the same
