import torch
import torch.nn as nn
from layer_norm import LayerNorm

class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by Layer Normalization.
    
    This is used in transformer architectures to stabilize training,
    improve gradient flow, and retain input information.

    Parameters:
    - features (int): The number of input features (same as model dimension).
    - dropout (float): Dropout probability to prevent overfitting.

    Shape:
    - Input: `x` -> [batch_size, seq_len, features]
    - Output: [batch_size, seq_len, features]
    
    
    What Does the ResidualConnection Layer Do?
        The ResidualConnection layer is crucial in transformers because it:

        Adds a residual (skip) connection → Helps gradient flow.
        Applies dropout → Prevents overfitting.
        Applies layer normalization → Stabilizes training.
        Why Is the Residual Connection Important?
        Prevents vanishing gradients

        Deep models (like transformers) suffer from gradients becoming too small.
        The residual connection allows gradients to flow directly through the network, improving training.
        > How Does the Residual Connection Help Gradient Flow?
            In deep neural networks, training can suffer from vanishing gradients—
            a problem where gradients become too small during backpropagation, making it hard 
            for the network to learn effectively. Residual connections solve this issue by allowing 
            gradients to flow directly through the network, preventing their decay.
            
        > 2️⃣ How Do Residual Connections Fix Gradient Flow?
            Residual connections shortcut the learning path by directly adding the input 𝑋 
            to the output of a sublayer F(X)
            
            Y = X + F(X)
            
            This simple addition ensures that:

            The input 𝑋 can flow directly through the network, even if the sublayers learn slowly.
            If F(X) is very small (e.g., early training stages), the network can still pass useful information forward.
            The gradient of the loss function bypasses deep layers and reaches earlier layers without shrinking too much.
            Gradient Flow Through a Residual Connection
            During backpropagation, we compute gradients using the chain rule. Let's see how residual connections impact this.

            For a standard deep network without residual connections:

            𝑌=𝐹(𝑋)

            The gradient of the loss L w.r.t. the input X is:
            dL/dX = dL/dY * dY/dX

            Since Y = F(X)

            dL/dX = dL/dY * F'(X)

            if  F'(X) is small (as in deep networks), the gradient shrinks significantly, leading to vanishing gradients.
            
            3️⃣ Why Is This Important in Transformers?
                Transformers are very deep networks (often with 12+ layers in BERT, 96+ in GPT-4). Without residual connections:

                Gradients would vanish quickly in backpropagation.
                Earlier layers (like initial embeddings) would struggle to learn.
                The entire model would take much longer to train, or might not train at all.
                By adding residual connections around self-attention and feedforward layers, transformers ensure:

                Stable training: Information from early layers flows efficiently.
                Better gradient flow: The model learns effectively across all layers.
                Faster convergence: The network trains in fewer steps
            
                
        Helps retain information from earlier layers

        Instead of completely replacing the input, the network adds the new transformation to the original input.
        Speeds up convergence

        By skipping unnecessary transformations, the model trains faster.
        
    What Happens Without the Residual Connection?
        ❌ Training becomes unstable
        ❌ Gradients vanish, leading to poor learning
        ❌ Loss does not converge as easily
        ❌ Model struggles to retain previous information

        📌 Without residual connections, transformers wouldn’t be able to train effectively!
        
        
    How Does ResidualConnection Affect the Transformer Architecture?
        The Residual Connection is used in both the encoder and decoder to improve information flow.

        📌 In the Transformer Encoder Block
        Each encoder layer consists of: 1️⃣ Multi-Head Self-Attention
        2️⃣ FeedForward Network

    Residual Connections are used around both components:
        
        self-attn residual: X' = X + self_attn(x)
        Feed forward res: X'' = X' + FFN(X')
        
        ✔ Preserves input features while improving representation
        
    📌 In the Transformer Decoder Block
        Each decoder layer consists of: 1️⃣ Masked Multi-Head Self-Attention
        2️⃣ Cross-Attention (attends to encoder output)
        3️⃣ FeedForward Network

        Residual connections are applied around all three components:
        
        X' = X + Masked self_attn(x)
        X'' = X' + cross_attn(x', encoder output)
        X''' = X'' + FFN(X'')
        
        ✔ Allows decoder to refine representations while maintaining earlier information


    🚀 Conclusion: Why Is ResidualConnection Essential?
        ✅ Prevents vanishing gradients → Enables deep transformer training
        ✅ Retains past information → Improves feature learning
        ✅ Stabilizes training → Prevents exploding gradients
        ✅ Speeds up learning → Faster convergence

        Without residual connections, transformers wouldn’t work as efficiently! 🎯


        
    """

    def __init__(self, features: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Forward pass for the residual connection.

        - Normalizes the input `x`
        - Applies the sublayer (e.g., Self-Attention or FeedForward)
        - Adds the residual connection (x + transformed_x)
        
        Args:
        - x (Tensor): Input tensor.
        - sublayer (Module): A sublayer (e.g., Self-Attention or FeedForward).

        Returns:
        - Tensor: Transformed tensor with residual connection.
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))


if __name__ == "__main__":
    batch_size, seq_len, features = 2, 5, 512
    x = torch.randn(batch_size, seq_len, features)

    residual_connection = ResidualConnection(features, dropout=0.1)

    # Example sublayer (simple Linear transformation)
    sublayer = nn.Linear(features, features)

    output = residual_connection(x, sublayer)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert x.shape == output.shape  # Ensuring input-output shape consistency
