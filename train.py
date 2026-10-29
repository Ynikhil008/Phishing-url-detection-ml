# src/train.py

import pandas as pd
import numpy as np
import joblib
import os
import json
from data_loader import DataLoader
from feature_extractor import URLFeatureExtractor
from preprocessor import Preprocessor
from models.classical_model import ClassicalPhishingDetector
from models.deep_learning_model import DeepLearningTrainer


def train_classical_pipeline(use_demo=True):
    """
    Full classical ML training pipeline.
    """
    print("\n" + "="*60)
    print("CLASSICAL ML PIPELINE")
    print("="*60)

    # ── Load Data ──
    loader = DataLoader()

    if use_demo:
        df = loader.create_demo_dataset(n_samples=5000)
    else:
        df = loader.load_uci_dataset()

    train_df, val_df, test_df = loader.split_data(df)

    # ── Feature Extraction ──
    print("\nExtracting features...")
    extractor = URLFeatureExtractor()

    X_train = extractor.extract_batch(
        train_df['url'].tolist(), disable_dns=True
    )
    X_val = extractor.extract_batch(
        val_df['url'].tolist(), disable_dns=True
    )
    X_test = extractor.extract_batch(
        test_df['url'].tolist(), disable_dns=True
    )

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # ── Preprocessing ──
    preprocessor = Preprocessor(
        scaler_type='standard', balance_method='smote'
    )
    X_train_proc, y_train_bal = preprocessor.fit_transform(X_train, y_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    preprocessor.save()

    feature_names = X_train.columns.tolist()

    # ── Train Multiple Models ──
    results = {}

    for model_type in ['random_forest', 'xgboost', 'logistic']:
        print(f"\n{'─'*40}")
        print(f"Training: {model_type}")
        detector = ClassicalPhishingDetector(model_type=model_type)

        # Optimize hyperparameters
        detector.optimize_hyperparameters(
            X_train_proc, y_train_bal, n_trials=20
        )

        # Train
        detector.train(X_train_proc, y_train_bal)

        # Evaluate on test set
        metrics = detector.evaluate(X_test_proc, y_test, feature_names)
        results[model_type] = metrics

        # Save model
        detector.save(f'models/{model_type}.pkl')

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/classical_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Classical pipeline complete. Results saved.")
    return results


def train_deep_learning_pipeline(email_csv_path=None):
    """
    Full deep learning training pipeline for email text classification.
    """
    print("\n" + "="*60)
    print("DEEP LEARNING PIPELINE")
    print("="*60)

    # ── Load Email Dataset ──
    if email_csv_path and os.path.exists(email_csv_path):
        df = pd.read_csv(email_csv_path)
        # Expect columns: 'text', 'label'
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    else:
        # Synthetic email data for demonstration
        print("Using synthetic email data...")
        phishing_texts = [
            "Dear customer, your account has been suspended. "
            "Click here immediately to verify your credentials: "
            "[secure-login.xyz](http://secure-login.xyz/verify)",

            "URGENT: Unusual activity detected on your PayPal account. "
            "Confirm your identity now to avoid permanent suspension.",

            "You have won $1,000,000! Click the link to claim your prize. "
            "Act now before it expires.",

            "Your bank account requires verification. "
            "Please update your information immediately.",

            "Security alert: Someone tried to access your account. "
            "Reset your password immediately.",
        ] * 400

        legitimate_texts = [
            "Hi John, please find attached the meeting agenda for tomorrow. "
            "Let me know if you have any questions.",

            "Your order has been shipped and will arrive in 3-5 business days. "
            "Track your package using the link below.",

            "Thank you for subscribing to our newsletter. "
            "You can unsubscribe at any time.",

            "Reminder: Your appointment is scheduled for Thursday at 2 PM. "
            "Please reply to confirm.",

            "The quarterly report is now available. "
            "Please review before the team meeting.",
        ] * 400

        texts = phishing_texts + legitimate_texts
        labels = [1] * len(phishing_texts) + [0] * len(legitimate_texts)

        # Shuffle
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

    # Split
    split = int(0.8 * len(texts))
    val_split = int(0.9 * len(texts))

    train_texts = texts[:split]
    train_labels = labels[:split]
    val_texts = texts[split:val_split]
    val_labels = labels[split:val_split]
    test_texts = texts[val_split:]
    test_labels = labels[val_split:]

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | "
          f"Test: {len(test_texts)}")

    # ── Train ──
    trainer = DeepLearningTrainer(
        model_name='bert-base-uncased',
        epochs=5,
        batch_size=16,
        lr=2e-5,
        max_length=256
    )

    history = trainer.train(train_texts, train_labels,
                             val_texts, val_labels)

    # ── Final Test Evaluation ──
    from torch.utils.data import DataLoader as TorchLoader
    from models.deep_learning_model import PhishingEmailDataset

    test_ds = PhishingEmailDataset(
        test_texts, test_labels, trainer.tokenizer
    )
    test_loader = TorchLoader(test_ds, batch_size=16)

    test_metrics = trainer.evaluate(test_loader)
    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save history
    with open('results/dl_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history, test_metrics


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Train classical models
    classical_results = train_classical_pipeline(use_demo=True)

    # Train deep learning model (comment out if no GPU)
    # dl_history, dl_metrics = train_deep_learning_pipeline()
