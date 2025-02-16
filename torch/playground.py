## Defining Modules

from torch import nn
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        return self.linear(x)


model = Model()
model_no_grad = Model()
model_no_grad.load_state_dict(model.state_dict())

x = torch.randn(10, 512)
out = model(x)  # torch.Size([10, 10])
with torch.no_grad():
    out_no_grad = model_no_grad(x)

# True, none are part of the computational graph
print(
    "Output the same with grad vs no_grad?: ",
    torch.equal(out, out_no_grad)
) # yes, has no computational graph params


###### Backprop vs no backprop

import torch
import torch.nn as nn
import torch.optim as optim


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


def train_step(model, optimizer, loss_fn, x, target):
    """Update weights"""
    model.train()
    optimizer.zero_grad()

    output = model(x)  # forward pass
    loss = loss_fn(output, target)  # loss computation
    loss.backward()  # gradient computation
    optimizer.step()  # weight update
    
    # get output after forward call of the updated weights outside here
    return loss.item()


def eval_step(model, loss_fn, x, target):
    """Do not update weights"""
    model.eval()
    with torch.no_grad():
        output = model(x)
        loss = loss_fn(output, target)

    return loss.item(), output


model = Model()
model_no_grad = Model()
model_no_grad.load_state_dict(model.state_dict())

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(10, 512)
target = torch.randn(10, 10)

loss_value = train_step(
    model,
    optimizer,
    loss_fn,
    x,
    target,
)
loss_value_no_grad, out_no_grad = eval_step(
    model_no_grad,
    loss_fn,
    x,
    target,
)

# get output after weight update
out = model(x) 

print(f"Loss (with gradients): {loss_value:.4f}")
print(f"Loss (without gradients): {loss_value_no_grad:.4f}")
print(
    f"loss equal after one step?: ",
    (loss_value == loss_value_no_grad)
)  # True
print(
    f"output equal after one step?: ",
    torch.equal(out, out_no_grad)
)  # False

for step in range(5):
    loss_value = train_step(model, optimizer, loss_fn, x, target)  # model update weights
    loss_value_no_grad, _ = eval_step(model_no_grad, loss_fn, x, target)  # model_no_grad no update weights

    # already trained once, start with 1
    print(f"training {step+2}: {loss_value:.4f}, eval_mode: {loss_value_no_grad:.4f}")

out = model(x)
out_no_grad = model_no_grad(x)
print("After training multiple steps:", torch.equal(out, out_no_grad))  # False
