# Anomaly Detector - Credit Card Fraud

Detects fraudulent credit card transactions using unsupervised learning. The models only see normal transactions during training, then flag anything unusual at test time.

**[Live demo](https://huggingface.co/spaces/EricRistol/anomaly-detector)**

## Dataset

Credit Card Fraud dataset from Kaggle: 284,807 transactions, 0.17% are fraud. Training set contains only normal (non-fraud) transactions.

## Models tested

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| Isolation Forest | 0.206 | 0.265 | 0.232 | 0.953 |
| One-Class SVM | 0.056 | 0.847 | 0.105 | 0.945 |
| Local Outlier Factor | 0.000 | 0.000 | 0.000 | 0.712 |

**Isolation Forest** wins with the best balance of precision and recall. One-Class SVM catches most fraud but creates too many false alarms. LOF doesn't work well on this high-dimensional imbalanced data.

## How to run

Get the data first:
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in `data/`

Then:

```bash
pip install -r requirements.txt
python train.py         # fit all three models
python evaluate.py      # precision, recall, F1, ROC-AUC
python predict.py -m isolation_forest   # interactive demo
```

## Files

```
├── data.py              load, scale, and split
├── train.py             fit Isolation Forest, LOF, One-Class SVM
├── evaluate.py          compute metrics and confusion matrices
├── predict.py           interactive predictor
├── hf-space/            Gradio demo
└── data/                put creditcard.csv here
```

---

**[Live demo](https://huggingface.co/spaces/EricRistol/anomaly-detector)**
