
import os
import pickle
import argparse
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

AVAILABLE_MODELS = ["isolation_forest", "lof", "ocsvm"]


def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_single(model, features):
    X = np.array(features).reshape(1, -1)
    raw = model.predict(X)[0]
    label = "FRAUD" if raw == -1 else "NORMAL"

    score = None
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
    elif hasattr(model, "score_samples"):
        score = model.score_samples(X)[0]

    return label, score


def demo_with_test_data(model_name):
    """Pick random test samples and show predictions."""
    model = load_model(model_name)
    data = np.load(os.path.join(MODEL_DIR, "test_data.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    #pick 5 normal and 5 fraud examples
    normal_idx = np.where(y_test == 0)[0]
    fraud_idx  = np.where(y_test == 1)[0]

    rng = np.random.default_rng(42)
    sample_idx = np.concatenate([
        rng.choice(normal_idx, size=min(5, len(normal_idx)), replace=False),
        rng.choice(fraud_idx, size=min(5, len(fraud_idx)), replace=False),
    ])

    print(f"\nModel: {model_name}")
    print(f"{'True Label':<12} {'Prediction':<12} {'Score':>10}")
    print("-" * 36)

    for i in sample_idx:
        label, score = predict_single(model, X_test[i])
        true = "fraud" if y_test[i] == 1 else "normal"
        score_str = f"{score:.4f}" if score is not None else "n/a"
        marker = " ✓" if (label == "FRAUD") == (y_test[i] == 1) else " ✗"
        print(f"{true:<12} {label:<12} {score_str:>10}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection predictor")
    parser.add_argument(
        "--model", "-m",
        choices=AVAILABLE_MODELS,
        default="isolation_forest",
        help="which model to use (default: isolation_forest)",
    )
    args = parser.parse_args()
    demo_with_test_data(args.model)
