import streamlit as st
import joblib
import pandas as pd
import os
import sys

# Import your modules
sys.path.append(os.path.dirname(__file__))
from feature_extractor import URLFeatureExtractor
from preprocessor import Preprocessor

st.set_page_config(page_title="Phishing Detector", layout="centered")

st.title("🔐 Phishing URL Detection System")

# Load models
@st.cache_resource
def load_models():
    extractor = URLFeatureExtractor()
    preprocessor = Preprocessor()

    try:
        preprocessor.load('models/preprocessor.pkl')
        model = joblib.load('models/random_forest.pkl')
        return extractor, preprocessor, model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None

extractor, preprocessor, model = load_models()

# Input
url = st.text_input("Enter URL to check:")

if st.button("Check URL"):

    if not url:
        st.warning("⚠️ Please enter a URL")
    elif model is None:
        st.error("Model not loaded properly")
    else:
        try:
            # Extract features
            features = extractor.extract(url)
            X = pd.DataFrame([features])
            X_proc = preprocessor.transform(X)

            prob = model.predict_proba(X_proc)[0]
            pred = model.predict(X_proc)[0]

            # Display result
            if pred == 1:
                st.error("🚨 PHISHING WEBSITE")
            else:
                st.success("✅ LEGITIMATE WEBSITE")

            st.write(f"**Confidence:** {round(max(prob), 4)}")
            st.write(f"**Phishing Probability:** {round(prob[1], 4)}")

            # Signals
            st.subheader("⚠️ Suspicious Signals")

            signals = []
            if features.get('has_ip_address'):
                signals.append("Contains IP address")
            if features.get('is_shortener'):
                signals.append("Uses URL shortener")
            if features.get('suspicious_keyword_count', 0) > 0:
                signals.append("Suspicious keywords found")
            if features.get('domain_age_days', 365) < 90:
                signals.append("New domain")
            if not features.get('uses_https'):
                signals.append("No HTTPS")
            if features.get('num_subdomains', 0) > 3:
                signals.append("Too many subdomains")
            if features.get('brand_impersonation'):
                signals.append("Brand impersonation")

            if signals:
                for s in signals:
                    st.write("•", s)
            else:
                st.write("No major issues detected")

        except Exception as e:
            st.error(f"❌ Error: {e}")