# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from feature_extractor import URLFeatureExtractor
from preprocessor import Preprocessor

app = FastAPI(
    title="Phishing Detection API",
    description="Real-time phishing detection using ML and DL models",
    version="1.0.0"
)

# Load models at startup
extractor = URLFeatureExtractor()
preprocessor = Preprocessor()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    preprocessor.load('models/preprocessor.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    xgb_model = joblib.load('models/xgboost.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    rf_model = None
    xgb_model = None


class URLRequest(BaseModel):
    url: str
    model: Optional[str] = "random_forest"

class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: str
    model: Optional[str] = "bert"

class BatchURLRequest(BaseModel):
    urls: list[str]
    model: Optional[str] = "random_forest"


@app.get("/")
def root():
    return {
        "message": "Phishing Detection API",
        "endpoints": ["/predict/url", "/predict/email", "/predict/batch", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": rf_model is not None,
        "extractor_ready": True
    }


@app.post("/predict/url")
def predict_url(request: URLRequest):
    if not request.url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        # Extract features
        features = extractor.extract(request.url)
        import pandas as pd
        X = pd.DataFrame([features])
        X_proc = preprocessor.transform(X)

        # Select model
        model = rf_model if request.model == "random_forest" else xgb_model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prob = model.predict_proba(X_proc)[0]
        pred = model.predict(X_proc)[0]

        # Build explanation
        suspicious_signals = []
        if features.get('has_ip_address'):
            suspicious_signals.append("Contains IP address instead of domain")
        if features.get('is_shortener'):
            suspicious_signals.append("Uses URL shortener")
        if features.get('suspicious_keyword_count', 0) > 0:
            suspicious_signals.append("Contains suspicious keywords")
        if features.get('domain_age_days', 365) < 90:
            suspicious_signals.append("Recently registered domain")
        if not features.get('uses_https'):
            suspicious_signals.append("Does not use HTTPS")
        if features.get('num_subdomains', 0) > 3:
            suspicious_signals.append("Excessive subdomains")
        if features.get('brand_impersonation'):
            suspicious_signals.append("Possible brand impersonation detected")

        return {
            "url": request.url,
            "prediction": "PHISHING" if pred == 1 else "LEGITIMATE",
            "phishing_probability": round(float(prob[1]), 4),
            "legitimate_probability": round(float(prob[0]), 4),
            "confidence": round(float(max(prob)), 4),
            "risk_level": (
                "HIGH" if prob[1] > 0.8 else
                "MEDIUM" if prob[1] > 0.5 else
                "LOW"
            ),
            "suspicious_signals": suspicious_signals,
            "model_used": request.model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchURLRequest):
    if len(request.urls) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 URLs"
        )

    results = []
    for url in request.urls:
        try:
            single = predict_url(URLRequest(url=url, model=request.model))
            results.append(single)
        except:
            results.append({
                "url": url,
                "prediction": "ERROR",
                "error": "Feature extraction failed"
            })

    phishing_count = sum(1 for r in results if r.get('prediction') == 'PHISHING')

    return {
        "total": len(results),
        "phishing_detected": phishing_count,
        "legitimate": len(results) - phishing_count,
        "results": results
    }


@app.post("/predict/email")
def predict_email(request: EmailRequest):
    """
    Simple heuristic email analysis (DL model requires GPU server setup).
    """
    text = f"{request.subject} {request.body}".lower()

    phishing_keywords = [
        'verify', 'suspended', 'immediately', 'urgent', 'password',
        'click here', 'confirm your', 'account locked', 'unusual activity',
        'prize', 'won', 'claim now', 'limited time', 'act now',
        'update your', 'validate', 'credential'
    ]

    keyword_hits = [kw for kw in phishing_keywords if kw in text]
    keyword_score = len(keyword_hits) / len(phishing_keywords)

    # Check sender domain
    sender_suspicious = 0
    if '@' in request.sender:
        domain = request.sender.split('@')[-1]
        if any(x in domain for x in ['.xyz', '.tk', '.ml', '.ga', '.cf']):
            sender_suspicious = 1

    # URL count in body
    import re
    urls_in_body = re.findall(r'http[s]?://\S+', request.body)
    url_score = min(len(urls_in_body) / 5, 1.0)

    combined_score = (keyword_score * 0.5 + sender_suspicious * 0.3
                      + url_score * 0.2)

    return {
        "prediction": "PHISHING" if combined_score > 0.3 else "LEGITIMATE",
        "phishing_probability": round(combined_score, 4),
        "keyword_hits": keyword_hits,
        "urls_found": len(urls_in_body),
        "suspicious_sender": bool(sender_suspicious),
        "risk_level": (
            "HIGH" if combined_score > 0.6 else
            "MEDIUM" if combined_score > 0.3 else
            "LOW"
        )
    }
