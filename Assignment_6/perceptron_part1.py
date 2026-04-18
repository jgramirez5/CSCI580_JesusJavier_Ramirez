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
        y = df["y"].values.astype(int)
    else:
        X = df.iloc[:, 0:2].values.astype(float)
        y = df.iloc[:, 2].values.astype(int)

    return X, y


# ----------------------------
# Step activation
# ----------------------------
def step(z):
    return 1 if z >= 0 else 0


# ----------------------------
# Train heuristic perceptron
# ----------------------------
def train_perceptron(X, y, learning_rate=0.1, max_epochs=100, seed=42):
    np.random.seed(seed)

    # random weights and bias
    w = np.random.randn(2)
    b = np.random.randn()

    # store boundary after each update/epoch for plotting
    history = [(w.copy(), b)]

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            z = np.dot(w, X[i]) + b
            y_hat = step(z)
            error = y[i] - y_hat

            # heuristic update rule from assignment
            b = b + learning_rate * error
            w = w + learning_rate * error * X[i]

            total_error += abs(error)

        history.append((w.copy(), b))

        # stop early if perfectly classified
        if total_error == 0:
            print(f"Converged at epoch {epoch + 1}")
            break

    return w, b, history


# ----------------------------
# Plot decision boundary
# ----------------------------
def plot_boundary(ax, w, b, x_min, x_max, color="k", linestyle="-", linewidth=1, alpha=1.0, label=None):
    if abs(w[1]) < 1e-10:
        # vertical line: w1*x + b = 0 -> x = -b/w1
        if abs(w[0]) > 1e-10:
            x_val = -b / w[0]
            ax.axvline(x=x_val, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
        return

    x_vals = np.array([x_min, x_max])
    y_vals = -(w[0] * x_vals + b) / w[1]
    ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)


# ----------------------------
# Plot results
# ----------------------------
def plot_results(X, y, history, learning_rate):
    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter data
    class0 = y == 0
    class1 = y == 1
    ax.scatter(X[class0, 0], X[class0, 1], label="Class 0")
    ax.scatter(X[class1, 0], X[class1, 1], label="Class 1")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1

    # initial line in red
    w0, b0 = history[0]
    plot_boundary(ax, w0, b0, x_min, x_max, color="red", linestyle="-", linewidth=2, label="Initial boundary")

    # middle lines in dashed green
    if len(history) > 2:
        for w_mid, b_mid in history[1:-1]:
            plot_boundary(ax, w_mid, b_mid, x_min, x_max, color="green", linestyle="--", linewidth=1, alpha=0.5)

    # final line in black
    wf, bf = history[-1]
    plot_boundary(ax, wf, bf, x_min, x_max, color="black", linestyle="-", linewidth=2.5, label="Final boundary")

    ax.set_title(f"Part 1 - Heuristic Perceptron (lr={learning_rate})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    X, y = load_data("data.csv")

    learning_rate = 0.1
    max_epochs = 100

    w, b, history = train_perceptron(X, y, learning_rate=learning_rate, max_epochs=max_epochs)

    print("Final weights:", w)
    print("Final bias:", b)
    print("Iterations:", len(history) - 1)

    plot_results(X, y, history, learning_rate)


if __name__ == "__main__":
    main()
