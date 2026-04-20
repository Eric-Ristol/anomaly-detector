---
title: Anomaly Detector
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.32.0"
app_file: app.py
pinned: false
license: mit
---

# Credit Card Fraud Detector

Unsupervised anomaly detection comparing **Isolation Forest** vs **One-Class SVM** on the Credit Card Fraud dataset (284K transactions, 0.17% fraud).

Models are trained only on normal transactions — they learn what "normal" looks like and flag deviations as potential fraud.

## Features

- Classify random transactions and see predictions vs ground truth
- Visualize anomaly score distributions (normal vs fraud)
- Compare model behavior on the same data
