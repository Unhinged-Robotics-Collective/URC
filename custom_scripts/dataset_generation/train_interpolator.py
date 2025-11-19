import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def make_model(X: np.ndarray, y: np.ndarray, predictor = SVR(kernel="rbf", C=2.0, epsilon=0.05, gamma="scale")) -> Pipeline:
    # X: shape (n_samples, n_features), y: shape (n_samples,)
    assert X.shape[0] == y.shape[0]
    model = make_pipeline(
        StandardScaler(),
        predictor
    )
    model.fit(X, y)
    return model


def load_clean_data(file_path: str):
    data = np.genfromtxt(
        file_path,
        delimiter=",",
        invalid_raise=False
    )
    return data


def plot_data(data: np.ndarray):
    # Split into x (first column) and Y (the remaining 63)
    x = data[:, 0]
    Y = data[:, 1:]

    # Plot all 63 curves
    plt.figure(figsize=(10, 6))
    for i in range(Y.shape[1]):
        plt.scatter(x, Y[:, i], alpha=0.6, lw=1)

    plt.title("All 63 Features vs First Column")
    plt.xlabel("Column 0 (x-axis)")
    plt.ylabel("Other Columns (y-axis)")
    plt.grid(True, alpha=0.3)

    # Optionally limit lines or make one bold for visibility
    # plt.plot(x, Y[:, 0], color='black', lw=2)

    plt.tight_layout()
    plt.show()


def train_model(file_path: str):
    data = load_clean_data(file_path)
    X = data[:, 1:]
    y = data[:, 0]
    model = make_model(X, y)
    return model


if __name__ == "__main__":
    data = load_clean_data("2025-11-15-19-16-55-347367.csv")
    print(data.shape)
    X = data[:, 1:]
    y = data[:, 0]
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model = make_model(X, y, rf)
    x_new = np.random.rand(63)            # shape (63,)
    x_new = x_new.reshape(1, -1)          # shape (1, 63)

    y_pred = model.predict(x_new)
    print(y.min(), y.max())
    hist, edges = np.histogram(y, bins=10)
    print("bins:", edges)
    print("counts:", hist)

    print("Prediction:", y_pred[0])
    plot_data(data)
