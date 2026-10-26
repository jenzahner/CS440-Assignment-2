"""Image examples, predictions and training plots for Problem 3. Do not edit."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

PURPLE, AMBER, GREEN = "#6E1A81", "#F6AA1C", "#076838"
_COLORS = (PURPLE, GREEN, AMBER)
_CONFUSION_CMAP = LinearSegmentedColormap.from_list("fashion_purple", ["#F7F2F8", PURPLE])


def _series_style(index):
    return {"color": _COLORS[index % len(_COLORS)], "linewidth": 2,
            "linestyle": ("-", "--", ":", "-.")[(index // len(_COLORS)) % 4]}


def _style_axes(ax, grid_axis="y"):
    ax.title.set_color(PURPLE)
    ax.title.set_fontsize(10)
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6, alpha=0.6)


def _annotation_color(rgba):
    channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                for c in rgba[:3]]
    luminance = sum(c * w for c, w in zip(channels, (0.2126, 0.7152, 0.0722)))
    return "white" if luminance < 0.179 else "black"


def _scores(model, X):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return torch.cat([model(batch) for batch in X.split(1024)])
    finally:
        model.train(was_training)


def plot_data(data, split="train"):
    """Display one image from each of the ten classes."""
    X, y = data[f"X_{split}"], data[f"y_{split}"]
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for label, ax in enumerate(axes.flat):
        index = (y == label).nonzero()[0, 0]
        ax.imshow(X[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}: {data['class_names'][label]}", fontsize=10, color=PURPLE)
        ax.axis("off")
    fig.tight_layout()
    return axes


def plot_predictions(model, data, mistakes_only=False, n=12):
    """Display validation images with their true and predicted labels."""
    X, y = data["X_val"], data["y_val"]
    pred = _scores(model, X).argmax(dim=1)
    indices = (pred != y).nonzero().flatten() if mistakes_only else torch.arange(len(y))
    indices = indices[:n]
    cols = min(4, max(1, len(indices)))
    rows = max(1, (len(indices) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        actual, guessed = int(y[index]), int(pred[index])
        ax.imshow(X[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {data['class_names'][actual]}\nPredicted: {data['class_names'][guessed]}",
                     color=GREEN if actual == guessed else PURPLE, fontsize=9)
    if not len(indices):
        axes[0, 0].text(0.5, 0.5, "No misclassified validation images", ha="center")
    fig.tight_layout()
    return axes


def plot_confusion_matrix(model, data):
    """Rows are true classes; columns are predictions. Values are row percentages."""
    y = data["y_val"]
    pred = _scores(model, data["X_val"]).argmax(dim=1)
    counts = torch.bincount(10 * y + pred, minlength=100).reshape(10, 10)
    shares = 100 * counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(shares, cmap=_CONFUSION_CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(10), data["class_names"], rotation=45, ha="right")
    ax.set_yticks(range(10), data["class_names"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Validation confusion matrix (%)", color=PURPLE, fontsize=11)
    for row in range(10):
        for col in range(10):
            value = shares[row, col].item()
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8,
                    color=_annotation_color(image.cmap(image.norm(value))))
    fig.colorbar(image, ax=ax, label="Percentage within true class")
    fig.tight_layout()
    return ax


def plot_losses(losses, ax=None, title="Training loss"):
    """One list of losses per epoch, or a dict {name: list of losses}."""
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    curves = losses if isinstance(losses, dict) else {"loss": losses}
    for index, (name, values) in enumerate(curves.items()):
        ax.plot(range(1, len(values) + 1), values, label=name, **_series_style(index))
    ax.set_xlabel("epoch")
    ax.set_ylabel("average cross-entropy loss")
    ax.set_title(title, fontsize=10)
    _style_axes(ax)
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    ax.figure.tight_layout()
    return ax


def plot_activations():
    """The activation functions available in the MLP and their derivatives."""
    x = torch.linspace(-5, 5, 401, requires_grad=True)
    funcs = {"relu": torch.relu, "leaky_relu": lambda t: torch.nn.functional.leaky_relu(t, 0.1),
             "tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": lambda t: t}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for index, (name, f) in enumerate(funcs.items()):
        y = f(x)
        (grad,) = torch.autograd.grad(y.sum(), x)
        axes[0].plot(x.detach(), y.detach(), label=name, **_series_style(index))
        axes[1].plot(x.detach(), grad, label=name, **_series_style(index))
    axes[0].set_title("activation", fontsize=10)
    axes[1].set_title("derivative", fontsize=10)
    axes[0].set_ylim(-1.5, 3)
    for ax in axes:
        _style_axes(ax)
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    axes[0].text(0.02, 0.02, "leaky_relu is drawn with slope 0.1 for visibility",
                 transform=axes[0].transAxes, fontsize=7)
    fig.tight_layout()
    return axes


def compare_models(MLP, train_model, configs, data, epochs=8, lr=0.001, batch_size=256, seed=0):
    """Train each config and plot validation accuracy and training loss.

    Each config gives hidden_sizes and activation, and may override epochs, lr
    or batch_size. Returns (config, model, losses, validation accuracy) per model.
    """
    if not configs:
        return []
    results, names = [], []
    for config in configs:
        torch.manual_seed(seed)
        model = MLP(784, config["hidden_sizes"], 10, config.get("activation", "relu"))
        losses = train_model(model, data["X_train"], data["y_train"], config.get("epochs", epochs),
                             config.get("lr", lr), config.get("batch_size", batch_size), seed)
        pred = _scores(model, data["X_val"]).argmax(dim=1)
        acc = (pred == data["y_val"]).float().mean().item()
        results.append((config, model, losses, acc))
        name = f"{config['hidden_sizes']} {config.get('activation', 'relu')}"
        names.append(name)
        print(f"{name}: validation accuracy {acc:.4f}")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    values = [r[3] for r in results]
    ax.barh(range(len(results)), values,
            color=[_series_style(i)["color"] for i in range(len(results))], height=0.65)
    ax.set_yticks(range(len(results)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation accuracy")
    _style_axes(ax, grid_axis="x")
    for i, value in enumerate(values):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    plot_losses({f"{i + 1}: {name}": r[2] for i, (name, r) in enumerate(zip(names, results))})
    return results
