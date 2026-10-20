# %% Task 1 cell 8
class ManualLinear:
    def __init__(self, inputs, outputs, dtype=torch.float32, device="cpu"):
        self.weight = torch.randn(inputs, outputs, dtype=dtype, device=device) * (2 / inputs) ** 0.5
        self.bias = torch.zeros(outputs, dtype=dtype, device=device)
    def forward(self, x):
        self.x = x
        return x @ self.weight + self.bias
    def backward(self, gradient):
        self.dw = self.x.T @ gradient
        self.db = gradient.sum(0)
        return gradient @ self.weight.T

class ManualReLU:
    def forward(self, x):
        self.mask = x > 0
        return x.clamp_min(0)
    def backward(self, gradient):
        return gradient * self.mask

class ManualSigmoid:
    def forward(self, x):
        self.output = torch.sigmoid(x)
        return self.output
    def backward(self, gradient):
        return gradient * self.output * (1 - self.output)

class ManualCrossEntropy:
    def forward(self, logits, labels):
        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        self.probabilities = log_probabilities.exp()
        self.labels = labels
        return -log_probabilities[torch.arange(len(labels), device=labels.device), labels].mean()
    def backward(self):
        gradient = self.probabilities.clone()
        gradient[torch.arange(len(self.labels), device=self.labels.device), self.labels] -= 1
        return gradient / len(self.labels)

class ManualMLP:
    def __init__(self, widths, activation="relu", dtype=torch.float32, device="cpu"):
        self.layers = []
        for i, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
            self.layers.append(ManualLinear(inputs, outputs, dtype, device))
            if i < len(widths) - 2:
                self.layers.append(ManualReLU() if activation == "relu" else ManualSigmoid())
        self.criterion = ManualCrossEntropy()
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def loss(self, x, y):
        return self.criterion.forward(self.forward(x), y)
    def backward(self):
        gradient = self.criterion.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    def linears(self):
        return [layer for layer in self.layers if isinstance(layer, ManualLinear)]
    def parameters_and_gradients(self):
        return [(p, g) for layer in self.linears()
                for p, g in [(layer.weight, layer.dw), (layer.bias, layer.db)]]

print("Manual layers ready; no autograd is used by these components.")


# %% Task 1 cell 10
seed_everything()
check_model = ManualMLP([4, 5, 3], dtype=torch.float64)
check_x = torch.randn(3, 4, dtype=torch.float64)
check_y = torch.tensor([0, 1, 2])
manual_loss = check_model.loss(check_x, check_y)
manual_dx = check_model.backward().clone()
pairs = check_model.parameters_and_gradients()
manual_gradients = [g.clone() for _, g in pairs] + [manual_dx]
reference_parameters = [p.clone().requires_grad_(True) for p, _ in pairs]
reference_x = check_x.clone().requires_grad_(True)
hidden = torch.relu(reference_x @ reference_parameters[0] + reference_parameters[1])
reference_loss = nn.functional.cross_entropy(
    hidden @ reference_parameters[2] + reference_parameters[3], check_y)
reference_loss.backward()
autograd_gradients = [p.grad for p in reference_parameters] + [reference_x.grad]

def relative_error(a, b):
    return ((a-b).norm() / (a.norm()+b.norm()).clamp_min(1e-15)).item()

step = 1e-6
names = ["W1", "b1", "W2", "b2", "input"]
print("Relative error = ||a-b|| / (||a|| + ||b||); h=1e-6")
print(f"{'Tensor':8s} {'Autograd error':>16s} {'Finite-diff error':>18s}  allclose(auto, FD)")
for name, tensor, manual, reference in zip(
        names, [p for p, _ in pairs] + [check_x], manual_gradients, autograd_gradients):
    numerical = torch.zeros_like(tensor)
    for j in range(tensor.numel()):
        flat = tensor.view(-1)
        original = flat[j].item()
        flat[j] = original + step
        plus = check_model.loss(check_x, check_y).item()
        flat[j] = original - step
        minus = check_model.loss(check_x, check_y).item()
        flat[j] = original
        numerical.view(-1)[j] = (plus - minus) / (2 * step)
    auto_ok = torch.allclose(manual, reference, rtol=1e-5, atol=1e-7)
    fd_ok = torch.allclose(manual, numerical, rtol=1e-5, atol=1e-7)
    print(f"{name:8s} {relative_error(manual, reference):16.3e} "
          f"{relative_error(manual, numerical):18.3e}  {auto_ok}, {fd_ok}")
    assert auto_ok and fd_ok, f"Gradient mismatch: {name}"


# %% Task 1 cell 12
seed_everything()
MLP_EPOCHS = 20
MLP_LR = 0.01
manual_model = ManualMLP([3072, 256, 10], device=DEVICE)
manual_train_loader = make_loader(train_dataset, shuffle=True)
manual_history = []

@torch.no_grad()
def evaluate_manual(model, loader):
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE).flatten(1), y.to(DEVICE)
        logits = model.forward(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count}

for epoch in range(1, MLP_EPOCHS + 1):
    loss_sum, count = 0.0, 0
    for x, y in manual_train_loader:
        x, y = x.to(DEVICE).flatten(1), y.to(DEVICE)
        loss = manual_model.loss(x, y)
        manual_model.backward()
        for parameter, gradient in manual_model.parameters_and_gradients():
            parameter.add_(gradient, alpha=-MLP_LR)
        loss_sum += loss.item() * len(y)
        count += len(y)
    metrics = evaluate_manual(manual_model, val_loader)
    manual_history.append({"epoch": epoch, "train_loss": loss_sum / count,
                           "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]})
    print(f"Epoch {epoch:02d}/{MLP_EPOCHS}: loss={loss_sum/count:.4f}, "
          f"validation accuracy={metrics['accuracy']:.2%}")
plot_history(manual_history, "Task 1(c): manual MLP")
save_results("task1_mlp", {"widths": [3072, 256, 10], "lr": MLP_LR,
                           "epochs": MLP_EPOCHS, "batch_size": BATCH_SIZE}, manual_history)
print(f"Final validation accuracy: {manual_history[-1]['val_accuracy']:.2%}")


# %% Task 1 cell 14
# Ten linear layers: nine hidden layers and one output layer.
# Identical weights and Kaiming-normal initialization isolate activation effects.
seed_everything()
gradient_x, gradient_y = next(iter(make_loader(train_dataset)))
gradient_x = gradient_x.to(DEVICE).flatten(1)
gradient_y = gradient_y.to(DEVICE)
widths = [3072] + [128] * 9 + [10]
gradient_results = {}
for activation in ["sigmoid", "relu"]:
    seed_everything()
    network = ManualMLP(widths, activation=activation, device=DEVICE)
    network.loss(gradient_x, gradient_y)
    network.backward()
    norms = [layer.dw.norm().item() for layer in network.linears()]
    gradient_results[activation] = norms
    print(activation, "weight-gradient norms:", [f"{v:.3e}" for v in norms])
    plt.semilogy(range(1, 11), np.maximum(norms, 1e-30), marker="o", label=activation)
plt.xticks(range(1, 11))
plt.xlabel("Linear layer (input to output)")
plt.ylabel("Weight-gradient L2 norm (log scale)")
plt.title("Task 1(d): gradient norms at initialization")
plt.legend()
plt.tight_layout()
plt.show()
(OUTPUT_DIR / "task1_gradient_norms.json").write_text(json.dumps(gradient_results, indent=2))
