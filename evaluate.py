# src/evaluate.py

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison(results_path='results/classical_results.json'):
    with open(results_path) as f:
        results = json.load(f)

    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        vals = [results[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Phishing Detection Model Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    plt.show()


def plot_training_history(history_path='results/dl_history.json'):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['val_f1'], 'g-o', label='Val F1')
    axes[1].set_title('Validation F1 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_model_comparison()
