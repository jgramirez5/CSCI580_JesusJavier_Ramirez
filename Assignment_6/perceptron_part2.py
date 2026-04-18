import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Load data
# ----------------------------
def load_data(csv_path="data.csv"):
    df = pd.read_csv(csv_path)

    # Supports either named columns or plain 3-column CSV
    if {"x1", "x2", "y"}.issubset(df.columns):
        X = df[["x1", "x2"]].values.astype(float)
        y = df["y"].values.astype(float)
    else:
        X = df.iloc[:, 0:2].values.astype(float)
        y = df.iloc[:, 2].values.astype(float)

    return X, y


# ----------------------------
# Sigmoid
# ----------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------
# Log loss
# ----------------------------
def log_loss(y_true, y_pred):
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ----------------------------
# Train neuron with gradient descent
# ----------------------------
def train_neuron(X, y, learning_rate=0.1, epochs=100, seed=42):
    np.random.seed(seed)

    # random weights and bias
    w = np.random.randn(2)
    b = np.random.randn()

    # history for plotting boundaries
    boundary_history = [(w.copy(), b)]

    # loss values every 10 epochs
    loss_history = []
    epoch_marks = []

    n = len(X)

    for epoch in range(1, epochs + 1):
        # forward pass
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        # gradients for binary cross-entropy with sigmoid
        dw = np.dot(X.T, (y_hat - y)) / n
        db = np.mean(y_hat - y)

        # update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        boundary_history.append((w.copy(), b))

        # compute and store log loss every 10 epochs
        if epoch % 10 == 0:
            z_now = np.dot(X, w) + b
            y_hat_now = sigmoid(z_now)
            loss = log_loss(y, y_hat_now)
            loss_history.append(loss)
            epoch_marks.append(epoch)
            print(f"Epoch {epoch}: loss = {loss:.6f}")

    return w, b, boundary_history, epoch_marks, loss_history


# ----------------------------
# Plot decision boundary
# ----------------------------
def plot_boundary(ax, w, b, x_min, x_max, color="k", linestyle="-", linewidth=1, alpha=1.0, label=None):
    if abs(w[1]) < 1e-10:
        if abs(w[0]) > 1e-10:
            x_val = -b / w[0]
            ax.axvline(x=x_val, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
        return

    x_vals = np.array([x_min, x_max])
    y_vals = -(w[0] * x_vals + b) / w[1]
    ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)


# ----------------------------
# Plot decision boundaries
# ----------------------------
def plot_solution_boundary(X, y, history, learning_rate, epochs):
    fig, ax = plt.subplots(figsize=(8, 6))

    class0 = y == 0
    class1 = y == 1
    ax.scatter(X[class0, 0], X[class0, 1], label="Class 0")
    ax.scatter(X[class1, 0], X[class1, 1], label="Class 1")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1

    # initial line in red
    w0, b0 = history[0]
    plot_boundary(ax, w0, b0, x_min, x_max, color="red", linestyle="-", linewidth=2, label="Initial boundary")

    # intermediate lines in dashed green
    if len(history) > 2:
        for w_mid, b_mid in history[1:-1]:
            plot_boundary(ax, w_mid, b_mid, x_min, x_max, color="green", linestyle="--", linewidth=1, alpha=0.4)

    # final line in black
    wf, bf = history[-1]
    plot_boundary(ax, wf, bf, x_min, x_max, color="black", linestyle="-", linewidth=2.5, label="Final boundary")

    ax.set_title(f"Part 2 - Neuron / Gradient Descent (lr={learning_rate}, epochs={epochs})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Plot error curve
# ----------------------------
def plot_error_curve(epoch_marks, loss_history, learning_rate, epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_marks, loss_history, marker="o")
    plt.title(f"Log Loss Every 10 Epochs (lr={learning_rate}, epochs={epochs})")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    X, y = load_data("data.csv")

    learning_rate = 0.1
    epochs = 100

    w, b, boundary_history, epoch_marks, loss_history = train_neuron(
        X, y, learning_rate=learning_rate, epochs=epochs
    )

    print("Final weights:", w)
    print("Final bias:", b)

    plot_solution_boundary(X, y, boundary_history, learning_rate, epochs)
    plot_error_curve(epoch_marks, loss_history, learning_rate, epochs)


if __name__ == "__main__":
    main()


