import torch
import matplotlib.pyplot as plt


class LinearRegressionModel(torch.nn.Module):
    """
    1. Choose model                         -> Linear regression   -> y = mx + b
    2. Choose loss criteria                 -> MSE                 -> 1 / n * (y - y_hat) ^ 2
    3. Choose params w* to minimize loss    -> m and b             -> gradients to min loss (w in argmin_w L(w) -> m and b). d(MSE)/dm, d(MSE)/db

    Define parameters
    grad_m = grad_m - d(MSE) / dm
    grad_b = grad_b - d(MSE) / db

    Chain rule with respect to m:
    1 / n * (y - (mx + b))^2
        m -> mx + b -> -1/n * u^2
        1 -> x -> -2/n
        combine: 1 * x * -2/n = -2x/n

    grad_m = -2x * (1/n) * (y - y_hat)
    grad_b = -2 * (1/n) * (y - y_hat)

    """

    def __init__(self, n_features: int):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.randn(n_features, 1))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        return x @ self.weight + self.bias

    def compute_loss_m(self, x, y, y_pred):
        error = y - y_pred
        grad_m = -2 * (x.T @ error) / x.size(0)
        return grad_m

    def compute_loss_b(self, y, y_pred):
        error = y - y_pred
        grad_b = -2 * torch.sum(error) / y.size(0)
        return grad_b

    def train(self, data, labels, lr=0.01, epochs=1000):
        for epoch in range(epochs):
            y_pred = self(data)
            grad_m = self.compute_loss_m(data, labels, y_pred)
            grad_b = self.compute_loss_b(labels, y_pred)

            with torch.no_grad():
                self.weight -= lr * grad_m
                self.bias -= lr * grad_b

            if (epoch + 1) % 100 == 0:
                loss = torch.mean((labels - y_pred) ** 2)
                print(f"[{epoch + 1}/{epochs}]: Loss: {loss.item():.4f}")

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


SEED = 42
torch.manual_seed(SEED)

n_features = 1
n_samples = 100
X = 2 * torch.rand(n_samples, n_features)

true_weights = torch.rand(n_features, 1) * 5
true_bias = 4.0
noise = torch.randn(100, 1)
y = X @ true_weights + true_bias + noise

model = LinearRegressionModel(n_features)
model.train(X, y, lr=0.01, epochs=1_000)

print("Learned parameters (weights and bias):")
print(
    f"True weight: {true_weights.numpy().ravel()}, Weight: {model.weight.data.numpy().ravel()}"
)
print(f"True bias: {true_bias}, Bias: {model.bias.data.numpy()}")


if n_features == 1:
    X_line = torch.linspace(0, 2, 100).unsqueeze(1)
    y_line = model.predict(X_line)

    plt.scatter(X.numpy(), y.numpy(), color="blue", label="Data")
    plt.plot(X_line.numpy(), y_line.numpy(), color="red", label="Regression Line")
    plt.xlabel("Feature 0")
    plt.ylabel("y")
    plt.legend()
    plt.show()
