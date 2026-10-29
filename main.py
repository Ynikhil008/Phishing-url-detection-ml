# main.py

import os
import sys
import argparse

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"
os.environ["QT_FONT_DPI"] = "96"

def main():
    parser = argparse.ArgumentParser(description='Phishing Detection System')
    parser.add_argument('--mode', choices=['train', 'api', 'ui', 'demo'],
                        default='demo')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV dataset')
    parser.add_argument('--url', type=str, default=None,
                        help='Single URL to analyze')

    args = parser.parse_args()

    if args.mode == 'train':
        sys.path.insert(0, 'src')
        from src.train import train_classical_pipeline
        train_classical_pipeline(use_demo=(args.data is None))

    elif args.mode == 'api':
        import uvicorn
        uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

    elif args.mode == 'ui':
        os.system("streamlit run streamlit_app.py")

    elif args.mode == 'demo':
        # Quick demo: analyze a few URLs
        sys.path.insert(0, 'src')
        from src.feature_extractor import URLFeatureExtractor

        test_urls = [
            "[google.com](https://www.google.com)",
            "[paypa1-secure-login.xyz](http://paypa1-secure-login.xyz/verify?user=admin)",
            "[github.com](https://github.com/openai)",
            "[192.168.1.1](http://192.168.1.1/login.php?redirect=bank)",
            "[bit.ly](http://bit.ly/3xPhish1ng-account-verify)",
        ]

        extractor = URLFeatureExtractor()
        print("\n" + "="*60)
        print("DEMO: URL Feature Analysis")
        print("="*60)

        for url in test_urls:
            features = extractor.extract(url)
            risk_score = (
                features.get('has_ip_address', 0) * 0.3 +
                features.get('is_shortener', 0) * 0.2 +
                min(features.get('suspicious_keyword_count', 0) / 5, 1) * 0.25 +
                (1 - features.get('uses_https', 0)) * 0.15 +
                features.get('brand_impersonation', 0) * 0.1
            )

            print(f"\nURL: {url[:70]}")
            print(f"  Risk Score : {risk_score:.2f}")
            print(f"  HTTPS      : {'Yes' if features['uses_https'] else 'No'}")
            print(f"  IP Address : {'Yes' if features['has_ip_address'] else 'No'}")
            print(f"  Shortener  : {'Yes' if features['is_shortener'] else 'No'}")
            print(f"  Keywords   : {features['suspicious_keyword_count']}")
            print(f"  Subdomains : {features['num_subdomains']}")
            print(f"  Entropy    : {features['domain_entropy']:.3f}")
            risk_label = "HIGH ⚠️" if risk_score > 0.5 else "LOW ✅"
            print(f"  Assessment : {risk_label}")


if __name__ == "__main__":
    main()
