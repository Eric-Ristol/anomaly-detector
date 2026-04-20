# Anomaly Detector — Credit Card Fraud

Unsupervised anomaly detection on the [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 0.17% fraud rate).

Models learn **only from normal transactions** — no fraud examples during training. At test time they flag anything that deviates from the learned "normal" envelope.

## Models compared

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.206 | 0.265 | 0.232 | 0.953 |
| One-Class SVM | 0.056 | 0.847 | 0.105 | 0.945 |
| Local Outlier Factor | 0.000 | 0.000 | 0.000 | 0.712 |

Key takeaway: Isolation Forest has the best balance (highest F1 and ROC-AUC). OCSVM catches most fraud (85% recall) but at the cost of many false alarms. LOF fails completely on this high-dimensional, extreme-imbalance task.

## Project structure

```
data.py       Load, scale, and split the dataset (train = normal only)
train.py      Fit Isolation Forest, LOF, and One-Class SVM
evaluate.py   Precision, recall, F1, ROC-AUC, confusion matrices
predict.py    Interactive predictor on random test samples
hf-space/     Gradio demo deployed on HuggingFace Spaces
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

## Run

```bash
python data.py          # verify data loads correctly
python train.py         # train all three models (~25s)
python evaluate.py      # full evaluation report
python predict.py -m isolation_forest   # interactive demo
```

## Live demo

[Try it on HuggingFace Spaces](https://huggingface.co/spaces/EricRistol/anomaly-detector)
