"""
hooks:

pro: fast testing, no code modification
cons: runtime overhead, hard to manage for large models, defined implicitly at runtime

"""

import torch
import torch.nn as nn
import torch.optim as optim

### hooks


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def print_hook(module, input, out):
    print(
        f"Forward Hook: {module.__class__.__name__}, ",
        f"processed input with shape {input[0].shape}, ",
        f"processed output with shape {out[0].shape}",
    )


model = Model()
hook = model.register_forward_hook(print_hook)

x = torch.randn(10, 512)
output = model(x)

hook.remove()


###################################################
### hooks content manager - target linear layers
###################################################

# storage
ACTIVATIONS_STORAGE = {}
GRADIENTS_STORAGE = {}


def forward_hook(module, input, output):
    ACTIVATIONS_STORAGE[module] = output


def backward_hook(module, grad_input, grad_output):
    GRADIENTS_STORAGE[module] = grad_output


class HookContextManager:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def __enter__(self):
        """
        hook linear layers

        usage:
            with HookContextManager(model) as ctx:

        """
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                self.hooks.append(layer.register_forward_hook(forward_hook))
                self.hooks.append(layer.register_full_backward_hook(backward_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()


model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(10, 512)

with HookContextManager(model):
    output = model(x)
    print("output shape: ", output.shape)

    target = torch.randn(output.shape)
    loss = loss_fn(output, target)
    loss.backward()

print("forward activations: ")
for layer, activation in ACTIVATIONS_STORAGE.items():
    print(f"{layer}: {activation.shape}")

print("\ngradients: ")
for layer, grad in GRADIENTS_STORAGE.items():
    print(f"{layer}: {grad[0].shape}")

"""
forward activations: 
Linear(in_features=512, out_features=256, bias=True): torch.Size([10, 256]) -> x: [10, 512], w: [512, 256]
Linear(in_features=256, out_features=10, bias=True): torch.Size([10, 10]) -> x: [10, 256], w: [256, 10]

gradients: 
Linear(in_features=256, out_features=10, bias=True): torch.Size([10, 10]) -> backprop of above
Linear(in_features=512, out_features=256, bias=True): torch.Size([10, 256])
"""
