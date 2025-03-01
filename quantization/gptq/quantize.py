"""
https://arxiv.org/pdf/2210.17323

1. Layer wise quantization

    Consider a linear layer in LLM. 
    y = w @ x + b

    each layer will be quantized independently to

    
    argmin_{w_hat} (w @ x - w_q @ x)^2   ....    (1)


2. Optimal Brain Quantization (OBQ)

    (1) can be rewritten as 
    Sigma(( wi * xi - w_qi * xi)^2 )
    
    so the sum of individial data. To obtain the quantized weight, each sample is 
    quantized independently, and the error from each sample is propagaged to other
    unquantized samples
    
    Pseudocode:
        - Quantize wi to w_qi 
        - Get the error, e = abs(wi - w_qi)         ... (5)
        - for j in range(i + 1, n), wj - alpha_i * e_i
        
    so 
        wi = wi - Sigma (alpha_j * error_j), for j > i (non-quantized weight)
    
    Why?
        Weight from a Tensor is selected by sampling and quantized. When one is quantized, it has to 
        propagate. Think of a see-saw, where if a row is quantized, the saw tips. Adjust to equilibrium 
        by propagating the error to the unquantized. 
    
    Effect:
        This reduces the quantization much less than if the errors are not propagated.
        
    ----------------------------------------------------
    
    From (1), w @ x yields the activation, the output of the layer. Therefore, minimizing (1) also
    means minimizing how weights affects the activations in the learned space.
    
    === Hessian ===
    
    Hessian = d^2 /(dW)^2 * E -> second derivative -- the rate of change. First derivative shows the direction of the change (gradient).
    
    This means that the hessian exposes how strongly the remaining weights are affected a small change in w_q.
    
    --- Derivation ---
    
    Start with (1), -> E = (w @ x - w_q @ x)^2. 
    d/dw * E = 2 * X.T * X * W - 2 * X.T * X * W_q (chain rule)             ... (2)
    H = d^2 /(dW)^2 * E = 2 * X.T * X, where X = X_F, the full, unquantized matrix
    
    the minimum/max is where the critical point is. Setting the derivative to 0 for (2), we get
    0 = 2 * X.T * X * W - 2 * X.T * X * W_q, 
    X.T * X * W  = X.T * X * W_q            ... (3)
    0 =  X.T * X * W_q - (X.T * X)^-1 *W
    0 = (W_q - W) X.T * X (X.T * X)^-1      ... (4)
    
    0 is if the error is none aka, no gradient. error = 0 = ...
    
    

    
    
    
    
    
    
    
    We also have 
    error = W - W_q
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




"""