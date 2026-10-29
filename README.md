# Phishing URL Detection Using Machine Learning

## Overview
A machine learning project designed to classify URLs as legitimate or phishing by analyzing URL-based features. The project compares Logistic Regression, Random Forest, and XGBoost and visualizes model performance.

## Models
- Logistic Regression
- Random Forest
- XGBoost

## Key Features
- URL feature extraction
- Phishing risk analysis
- ROC curve and AUC evaluation
- Confusion matrix visualization
- Feature-importance analysis

## Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, and FastAPI.

## Results
The current evaluation plots show an AUC of 1.00 for the displayed models and 1,000 correct predictions out of 1,000 samples in each confusion matrix. These results should be independently verified using a leakage-free test set before being treated as generalizable performance.

## Project Structure
- `main.py` — command-line entry point
- `requirements.txt` — Python dependencies
- `src/` — training, API, and feature-extraction modules
- `results/` — evaluation plots

## Setup
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the demo:
```bash
python main.py --mode demo
```

Train the model:
```bash
python main.py --mode train
```

Run the Streamlit interface:
```bash
python main.py --mode ui
```

Run the API:
```bash
python main.py --mode api
```

## Disclaimer
This project is for educational and research purposes. Model predictions should not be treated as a guarantee that a URL is safe or malicious.
