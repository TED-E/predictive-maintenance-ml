# 🤖 Predictive Maintenance ML

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A machine learning pipeline to predict machinery failures before they occur, using real-time sensor data and classification algorithms.

---

## 📌 Features

- 📊 **Data preprocessing** — handles missing values, outliers, and feature scaling
- 🤖 **Multiple ML models** — Random Forest, Gradient Boosting, Logistic Regression
- 🔍 **Model evaluation** — accuracy, precision, recall, F1-score, confusion matrix
- 💾 **Model persistence** — save and load trained models with joblib
- 📈 **Feature importance plots** — visualize what sensor readings matter most
- 🚀 **Prediction API** — simple interface to predict on new sensor readings

---

## 📁 Project Structure

```
predictive-maintenance-ml/
├── data/
│   └── sample_sensor_data.csv
├── models/
│   └── trained_model.pkl
├── src/
│   ├── preprocess.py     # Data cleaning and feature engineering
│   ├── train.py          # Model training and evaluation
│   └── predict.py        # Inference on new data
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/TED-E/predictive-maintenance-ml.git
cd predictive-maintenance-ml
pip install -r requirements.txt
python src/train.py
```

---

## 💻 Usage

### Train the model
```python
from src.train import MaintenanceModel

model = MaintenanceModel(data_path='data/sample_sensor_data.csv')
model.train()
model.evaluate()
model.save('models/trained_model.pkl')
```

### Predict failures
```python
from src.predict import predict_failure

# sensor_readings: dict of feature_name -> value
readings = {
    'temperature': 85.3,
    'vibration': 0.72,
    'pressure': 102.5,
    'rpm': 1480,
    'current_draw': 14.2
}
result = predict_failure('models/trained_model.pkl', readings)
print(f"Failure predicted: {result['failure']} | Confidence: {result['confidence']:.1%}")
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Random Forest | 94.2% | 93.8% | 91.5% | 92.6% |
| Gradient Boosting | 93.7% | 92.4% | 90.8% | 91.6% |
| Logistic Regression | 87.3% | 86.1% | 84.7% | 85.4% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| scikit-learn | ML models and evaluation |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualization |
| joblib | Model serialization |

---

## 📄 License

MIT License © 2026 [TED-E](https://github.com/TED-E)
