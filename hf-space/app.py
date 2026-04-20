"""
Anomaly Detector – Gradio demo
Compares Isolation Forest vs One-Class SVM on credit card fraud detection.
Users can explore random transactions, see anomaly scores, and visualize
the score distributions for normal vs fraud transactions.
"""

import os
import pickle
import numpy as np
import gradio as gr

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── load models and data at startup ───────────────────────────────────
def load_pkl(name):
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

iso_forest = load_pkl("isolation_forest")
ocsvm      = load_pkl("ocsvm")

data = np.load(os.path.join(MODEL_DIR, "test_sample.npz"))
X_test, y_test = data["X_test"], data["y_test"]

MODELS = {
    "Isolation Forest": iso_forest,
    "One-Class SVM": ocsvm,
}

rng = np.random.default_rng()


# ── helpers ───────────────────────────────────────────────────────────
def get_score(model, X):
    if hasattr(model, "decision_function"):
        return -model.decision_function(X)
    return np.zeros(len(X))


def classify_random(model_name, n_samples):
    """Pick random test samples and classify them."""
    n = int(n_samples)
    model = MODELS[model_name]

    idx = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    X_sample = X_test[idx]
    y_true = y_test[idx]

    raw_preds = model.predict(X_sample)
    y_pred = (raw_preds == -1).astype(int)
    scores = get_score(model, X_sample)

    rows = []
    for i in range(len(idx)):
        true_label = "FRAUD" if y_true[i] == 1 else "Normal"
        pred_label = "FRAUD" if y_pred[i] == 1 else "Normal"
        correct = "Yes" if y_true[i] == y_pred[i] else "No"
        rows.append([true_label, pred_label, f"{scores[i]:.4f}", correct])

    #compute summary stats
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    fraud_in_sample = int(y_true.sum())
    detected = tp

    summary = (
        f"**{model_name}** on {n} random transactions\n\n"
        f"Fraud in sample: {fraud_in_sample} | "
        f"Detected: {detected} | "
        f"Missed: {fn} | "
        f"False alarms: {fp}\n\n"
        f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}"
    )

    return rows, summary


def score_distribution(model_name):
    """Compute anomaly scores for all test data, return as plot data."""
    model = MODELS[model_name]
    scores = get_score(model, X_test)

    normal_scores = scores[y_test == 0]
    fraud_scores  = scores[y_test == 1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_scores, bins=80, alpha=0.6, label="Normal", color="#3b82f6", density=True)
    ax.hist(fraud_scores, bins=30, alpha=0.7, label="Fraud", color="#ef4444", density=True)
    ax.set_xlabel("Anomaly Score (higher = more suspicious)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Score Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="Anomaly Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔍 Credit Card Fraud Detector\n"
        "Unsupervised anomaly detection on the Credit Card Fraud dataset. "
        "Models are trained **only on normal transactions** — they learn what "
        "'normal' looks like and flag anything unusual.\n\n"
        "*Compare Isolation Forest (tree-based) vs One-Class SVM (boundary-based) "
        "on real transaction data with 0.17% fraud rate.*"
    )

    with gr.Tab("Classify Transactions"):
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Isolation Forest",
                label="Model",
            )
            n_slider = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="Number of random transactions",
            )
            run_btn = gr.Button("Classify", variant="primary")

        summary_md = gr.Markdown()
        results_table = gr.Dataframe(
            headers=["True Label", "Prediction", "Anomaly Score", "Correct?"],
            label="Results",
        )
        run_btn.click(
            fn=classify_random,
            inputs=[model_dd, n_slider],
            outputs=[results_table, summary_md],
        )

    with gr.Tab("Score Distribution"):
        gr.Markdown(
            "How well does the model separate normal from fraud? "
            "The further apart the two distributions, the better."
        )
        model_dd2 = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="Isolation Forest",
            label="Model",
        )
        plot_btn = gr.Button("Show Distribution", variant="primary")
        plot_out = gr.Plot()
        plot_btn.click(
            fn=score_distribution,
            inputs=[model_dd2],
            outputs=[plot_out],
        )

demo.launch()
