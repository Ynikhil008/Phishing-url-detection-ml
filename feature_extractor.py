# src/feature_extractor.py

import re
import tldextract
import dns.resolver
import whois
import numpy as np
from urllib.parse import urlparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class URLFeatureExtractor:
    """
    Extracts 25+ engineered features from raw URLs.
    Covers lexical, DNS, WHOIS, and content-based signals.
    """

    SHORTENERS = {
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
        'short.link', 'buff.ly', 'rebrand.ly', 'cutt.ly'
    }

    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'verify', 'secure', 'account', 'update',
        'confirm', 'banking', 'password', 'credential', 'wallet',
        'support', 'alert', 'suspended', 'unusual', 'activity'
    ]

    def extract(self, url: str) -> dict:
        features = {}
        try:
            parsed = urlparse(url)
            ext = tldextract.extract(url)

            # --- Lexical Features ---
            features['url_length'] = len(url)
            features['domain_length'] = len(ext.domain)
            features['path_length'] = len(parsed.path)
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_question_marks'] = url.count('?')
            features['num_equals'] = url.count('=')
            features['num_at_symbols'] = url.count('@')
            features['num_ampersands'] = url.count('&')
            features['num_exclamations'] = url.count('!')
            features['num_spaces'] = url.count('%20')
            features['num_tilde'] = url.count('~')
            features['num_percent'] = url.count('%')

            # --- Structural Features ---
            features['uses_https'] = 1 if parsed.scheme == 'https' else 0
            features['has_ip_address'] = 1 if re.match(
                r'\d+\.\d+\.\d+\.\d+', parsed.netloc or ''
            ) else 0
            features['has_at_symbol'] = 1 if '@' in url else 0
            features['has_double_slash'] = 1 if '//' in parsed.path else 0
            features['has_prefix_suffix'] = 1 if '-' in ext.domain else 0
            features['is_shortener'] = 1 if any(
                s in url for s in self.SHORTENERS
            ) else 0

            # --- Subdomain Features ---
            subdomains = ext.subdomain.split('.') if ext.subdomain else []
            features['num_subdomains'] = len(subdomains)
            features['subdomain_length'] = len(ext.subdomain)

            # --- Entropy of Domain (random-looking domains score high) ---
            features['domain_entropy'] = self._entropy(ext.domain)

            # --- Keyword Features ---
            url_lower = url.lower()
            features['suspicious_keyword_count'] = sum(
                1 for kw in self.SUSPICIOUS_KEYWORDS if kw in url_lower
            )
            features['brand_impersonation'] = self._check_brand_impersonation(url_lower)

            # --- TLD Features ---
            features['tld_length'] = len(ext.suffix)
            features['is_common_tld'] = 1 if ext.suffix in [
                'com', 'org', 'net', 'edu', 'gov'
            ] else 0

            # --- Digit Features ---
            features['digit_ratio_url'] = sum(
                c.isdigit() for c in url
            ) / max(len(url), 1)
            features['digit_ratio_domain'] = sum(
                c.isdigit() for c in ext.domain
            ) / max(len(ext.domain), 1)

            # --- DNS Features (with fallback) ---
            dns_features = self._get_dns_features(ext.registered_domain or '')
            features.update(dns_features)

            # --- WHOIS Features (with fallback) ---
            whois_features = self._get_whois_features(ext.registered_domain or '')
            features.update(whois_features)

        except Exception as e:
            # Return zero features if extraction fails
            features = self._zero_features()

        return features

    def _entropy(self, s: str) -> float:
        """Shannon entropy of a string."""
        if not s:
            return 0.0
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in prob if p > 0)

    def _check_brand_impersonation(self, url: str) -> int:
        """
        Detects slight misspellings of major brands.
        e.g., 'paypa1', 'g00gle', 'arnazon'
        """
        brands = {
            'paypal': ['paypa1', 'paypai', 'paypa|'],
            'google': ['g00gle', 'gooogle', 'googie'],
            'amazon': ['arnazon', 'amazom', 'amaz0n'],
            'microsoft': ['micros0ft', 'micosoft', 'microzoft'],
            'apple': ['app1e', 'appl3', 'aple'],
            'facebook': ['faceb00k', 'facebok', 'facebo0k'],
            'netflix': ['netf1ix', 'netfl1x', 'neflix'],
        }
        for brand, fakes in brands.items():
            if any(f in url for f in fakes):
                return 1
            # Check if brand name appears in domain of a non-brand URL
            # e.g., paypal.suspicious.com
        return 0

    def _get_dns_features(self, domain: str) -> dict:
        features = {
            'dns_resolves': 0,
            'has_mx_record': 0,
            'dns_ttl': 0,
            'num_dns_records': 0
        }
        try:
            answers = dns.resolver.resolve(domain, 'A', lifetime=3)
            features['dns_resolves'] = 1
            features['num_dns_records'] = len(answers)
            features['dns_ttl'] = answers.rrset.ttl
        except:
            pass
        try:
            mx = dns.resolver.resolve(domain, 'MX', lifetime=3)
            features['has_mx_record'] = 1 if mx else 0
        except:
            pass
        return features

    def _get_whois_features(self, domain: str) -> dict:
        features = {
            'domain_age_days': -1,
            'domain_expiry_days': -1,
            'registrar_known': 0
        }
        try:
            w = whois.whois(domain)
            known_registrars = [
                'GoDaddy', 'Namecheap', 'Google', 'Cloudflare',
                'Network Solutions', 'MarkMonitor'
            ]
            if w.creation_date:
                cd = w.creation_date
                if isinstance(cd, list):
                    cd = cd[0]
                features['domain_age_days'] = (datetime.now() - cd).days
            if w.expiration_date:
                ed = w.expiration_date
                if isinstance(ed, list):
                    ed = ed[0]
                features['domain_expiry_days'] = (ed - datetime.now()).days
            if w.registrar:
                features['registrar_known'] = 1 if any(
                    r in w.registrar for r in known_registrars
                ) else 0
        except:
            pass
        return features

    def _zero_features(self) -> dict:
        """Returns a zero-filled feature dict as fallback."""
        return {k: 0 for k in [
            'url_length', 'domain_length', 'path_length', 'num_dots',
            'num_hyphens', 'num_underscores', 'num_slashes',
            'num_question_marks', 'num_equals', 'num_at_symbols',
            'num_ampersands', 'num_exclamations', 'num_spaces',
            'num_tilde', 'num_percent', 'uses_https', 'has_ip_address',
            'has_at_symbol', 'has_double_slash', 'has_prefix_suffix',
            'is_shortener', 'num_subdomains', 'subdomain_length',
            'domain_entropy', 'suspicious_keyword_count',
            'brand_impersonation', 'tld_length', 'is_common_tld',
            'digit_ratio_url', 'digit_ratio_domain', 'dns_resolves',
            'has_mx_record', 'dns_ttl', 'num_dns_records',
            'domain_age_days', 'domain_expiry_days', 'registrar_known'
        ]}

    def extract_batch(self, urls: list, disable_dns=False) -> pd.DataFrame:
        """
        Extract features for a list of URLs.
        Set disable_dns=True for faster offline processing.
        """
        if disable_dns:
            # Monkey-patch DNS/WHOIS methods to skip network calls
            self._get_dns_features = lambda d: {
                'dns_resolves': 0, 'has_mx_record': 0,
                'dns_ttl': 0, 'num_dns_records': 0
            }
            self._get_whois_features = lambda d: {
                'domain_age_days': -1, 'domain_expiry_days': -1,
                'registrar_known': 0
            }

        features = []
        for url in tqdm(urls, desc="Extracting URL features"):
            features.append(self.extract(url))
        return pd.DataFrame(features)
