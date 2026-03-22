"""
Predictive Maintenance ML - Training Pipeline
Author: TED-E
Description: Train and evaluate ML models to predict machinery failures.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class MaintenanceModel:
    """
    ML pipeline for predictive maintenance classification.

    Parameters
    ----------
    data_path : str
        Path to the CSV sensor dataset
    target_col : str
        Name of the target column (default: 'failure')
    model_type : str
        'random_forest', 'gradient_boosting', or 'logistic_regression'
    """

    MODELS = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    def __init__(self, data_path: str, target_col: str = 'failure',
                 model_type: str = 'random_forest'):
        self.data_path = data_path
        self.target_col = target_col
        self.model_type = model_type
        self.model = self.MODELS[model_type]
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.X_test = None
        self.y_test = None

    def load_and_split(self):
        """Load data, scale features, and split into train/test sets."""
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)
        self.feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test
        return X_train, y_train

    def train(self):
        """Train the selected model."""
        X_train, y_train = self.load_and_split()
        print(f"Training {self.model_type} on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self) -> dict:
        """Evaluate model on test set and print report."""
        y_pred = self.model.predict(self.X_test)
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
        }
        print("\n=== Model Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\n" + classification_report(self.y_test, y_pred))
        return metrics

    def plot_confusion_matrix(self, save_path: str = None):
        """Plot and optionally save confusion matrix."""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.model_type}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()

    def save(self, model_path: str):
        """Save model and scaler to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'features': self.feature_cols}, model_path)
        print(f"Model saved to {model_path}")


if __name__ == '__main__':
    m = MaintenanceModel('data/sample_sensor_data.csv', model_type='random_forest')
    m.train()
    m.evaluate()
    m.save('models/trained_model.pkl')
