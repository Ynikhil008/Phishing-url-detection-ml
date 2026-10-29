# src/data_loader.py

import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_uci_dataset(self):
        url = "[archive.ics.uci.edu](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff)"
        path = os.path.join(self.data_dir, "uci_phishing.arff")

        if not os.path.exists(path):
            print("Downloading UCI dataset...")
            r = requests.get(url)
            with open(path, 'wb') as f:
                f.write(r.content)

        rows = []
        data_section = False
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.lower() == '@data':
                    data_section = True
                    continue
                if data_section and line:
                    rows.append(line.split(','))

        columns = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service',
            'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
            'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
            'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
            'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
            'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
            'Google_Index', 'Links_pointing_to_page', 'Statistical_report', 'Result'
        ]

        df = pd.DataFrame(rows, columns=columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        df['label'] = df['Result'].apply(lambda x: 1 if x == 1 else 0)
        df.drop('Result', axis=1, inplace=True)

        print(f"UCI Dataset loaded: {df.shape}")
        print(df['label'].value_counts())
        return df

    def load_csv_dataset(self, path, url_col='url', label_col='label',
                         phishing_value=1):
        df = pd.read_csv(path)
        df = df[[url_col, label_col]].dropna()
        df.columns = ['url', 'label']
        df['label'] = df['label'].apply(
            lambda x: 1 if x == phishing_value else 0
        )
        print(f"CSV Dataset loaded: {df.shape}")
        print(df['label'].value_counts())
        return df

    def load_phishtank_feed(self, limit=5000):
        print("Loading PhishTank feed...")
        url = "[data.phishtank.com](http://data.phishtank.com/data/online-valid.json)"
        try:
            r = requests.get(url, timeout=30)
            data = r.json()[:limit]
            urls = [entry['url'] for entry in data]
            df = pd.DataFrame({'url': urls, 'label': 1})
            print(f"PhishTank loaded: {len(df)} phishing URLs")
            return df
        except Exception as e:
            print(f"PhishTank load failed: {e}")
            return pd.DataFrame(columns=['url', 'label'])

    def create_demo_dataset(self, n_samples=2000):
        np.random.seed(42)

        prefixes = ['secure', 'login', 'verify', 'update', 'account']
        suffixes = ['.xyz', '.tk', '.ml', '.ga', '.info']

        phishing_urls = []
        for _ in range(n_samples // 2):
            prefix = np.random.choice(prefixes)
            rand_str = ''.join(np.random.choice(list('abcdefghij'), 8))
            suffix = np.random.choice(suffixes)
            url = "http://" + prefix + "-" + rand_str + suffix + "/login"
            phishing_urls.append(url)

        legit_domains = ['google', 'amazon', 'microsoft', 'github', 'stackoverflow']
        legit_urls = []
        for _ in range(n_samples // 2):
            domain = np.random.choice(legit_domains)
            rand_path = ''.join(np.random.choice(list('abcdefghij'), 6))
            url = "[www](https://www)" + domain + ".com/" + rand_path
            legit_urls.append(url)

        df = pd.DataFrame({
            'url': phishing_urls + legit_urls,
            'label': [1] * (n_samples // 2) + [0] * (n_samples // 2)
        })

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Demo dataset created: {df.shape}")
        print(df['label'].value_counts())
        return df

    def split_data(self, df, test_size=0.2, val_size=0.1):
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=42,
            stratify=train_val['label']
        )
        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train, val, test
