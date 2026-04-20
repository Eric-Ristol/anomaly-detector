"""
train.py – Train three unsupervised anomaly detection models.

Each model learns only from normal transactions and flags anything
that deviates as an anomaly.

Models
------
1. Isolation Forest   – tree-based; isolates anomalies with fewer splits
2. Local Outlier Factor (LOF) – density-based; compares local reachability
3. One-Class SVM      – boundary-based; learns a tight envelope around normal data

All three output -1 for anomaly, 1 for normal. We convert to 1=fraud, 0=normal
to match the original labels for evaluation.
"""

import os
import time
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import data as data_module

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

#contamination = expected fraud ratio in the wild (~0.17%)
CONTAMINATION = 0.00173

MODELS = {
    "isolation_forest": IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    ),
    "lof": LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION,
        novelty=True,       #must be True to call predict() on new data
        n_jobs=-1,
    ),
    "ocsvm": OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.01,            #upper bound on fraction of outliers
    ),
}


def train_all():
    """Train every model on normal-only data and save to disk."""
    X_train, X_test, y_train, y_test, scaler, feature_cols = data_module.get_data()

    results = {}
    for name, model in MODELS.items():
        print(f"\nTraining {name} …")
        t0 = time.time()

        #OCSVM is O(n²~n³) — subsample to keep training under a minute
        if name == "ocsvm" and X_train.shape[0] > 30_000:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(X_train.shape[0], size=30_000, replace=False)
            model.fit(X_train[idx])
            print(f"  (subsampled to 30,000 for speed)")
        else:
            model.fit(X_train)

        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

        #save the fitted model
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  saved → {path}")

        results[name] = model

    #also save the scaler and feature columns for the demo
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    #save test data for evaluation
    np.savez(
        os.path.join(MODEL_DIR, "test_data.npz"),
        X_test=X_test, y_test=y_test,
    )
    print(f"\nTest data saved ({X_test.shape[0]:,} samples)")
    return results


if __name__ == "__main__":
    train_all()
