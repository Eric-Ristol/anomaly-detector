
import os
import pickle
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_test_data():
    data = np.load(os.path.join(MODEL_DIR, "test_data.npz"))
    return data["X_test"], data["y_test"]


def convert_predictions(raw_preds):
    """sklearn outputs -1=anomaly, 1=normal. Convert to 1=fraud, 0=normal."""
    return (raw_preds == -1).astype(int)


def get_scores(model, X, name):
    """Get anomaly scores (higher = more anomalous)."""
    if hasattr(model, "decision_function"):
        #Isolation Forest and OCSVM: lower = more anomalous, so negate
        return -model.decision_function(X)
    elif hasattr(model, "score_samples"):
        return -model.score_samples(X)
    return None


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a single model and print results."""
    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")

    raw_preds = model.predict(X_test)
    y_pred = convert_predictions(raw_preds)

    print(classification_report(
        y_test, y_pred, target_names=["Normal", "Fraud"], digits=4
    ))

    cm = confusion_matrix(y_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"  True Negatives : {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives : {tp:,}")

    scores = get_scores(model, X_test, name)
    if scores is not None:
        roc = roc_auc_score(y_test, scores)
        ap  = average_precision_score(y_test, scores)
        print(f"\n  ROC-AUC          : {roc:.4f}")
        print(f"  Avg Precision (PR): {ap:.4f}")

    return {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1":        2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def evaluate_all():
    """Load all models and evaluate on test data."""
    X_test, y_test = load_test_data()
    print(f"Test set: {len(y_test):,} samples "
          f"({(y_test==1).sum()} fraud, {(y_test==0).sum()} normal)")

    model_names = ["isolation_forest", "lof", "ocsvm"]
    results = {}
    for name in model_names:
        model = load_model(name)
        results[name] = evaluate_model(name, model, X_test, y_test)

    #summary comparison
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")


if __name__ == "__main__":
    evaluate_all()
