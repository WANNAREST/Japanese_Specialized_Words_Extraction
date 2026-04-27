# ml_engine.py
import os
import random
import re
from dataclasses import dataclass
from typing import List, Sequence, Set
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import JAPANESE_STOPWORDS
from text_utils import normalize_text
from nlp_processor import extract_candidates
@dataclass
class TermDiscoveryResult:
    candidate: str
    score: float
    is_known_dictionary_term: bool

def read_dictionary_terms(csv_path: str, column_name: str = "Japanese", limit: int = 8000) -> List[str]:
    df = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Cannot read CSV: {csv_path}")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV columns: {list(df.columns)}")

    terms = [normalize_text(v) for v in df[column_name].dropna().astype(str).tolist()]
    terms = [t for t in terms if len(t) >= 2]
    return list(dict.fromkeys(terms))[:limit]

def load_confirmed_terms(confirmed_terms_path: str) -> List[str]:
    if not confirmed_terms_path or not os.path.exists(confirmed_terms_path):
        return []
    with open(confirmed_terms_path, "r", encoding="utf-8") as f:
        lines = [normalize_text(line) for line in f.readlines()]
    return [x for x in lines if len(x) >= 2 and not x.startswith("#")]

def _derive_pseudo_negatives(positive_terms: Sequence[str], seed: int = 42) -> List[str]:
    random.seed(seed)
    negatives: Set[str] = set(JAPANESE_STOPWORDS)

    for term in positive_terms:
        clean = re.sub(r"[・/／,，.．・;；:：]", " ", term)
        pieces = [p for p in clean.split() if p]
        for p in pieces:
            if len(p) == 1: negatives.add(p)
            if 2 <= len(p) <= 3: negatives.add(p)
        if len(term) >= 6:
            negatives.add(term[:2])
            negatives.add(term[-2:])

    negatives.update({"2024", "123", "abc", "www", "http", "ver2", "pdf", "xlsx"})
    negatives = {normalize_text(n) for n in negatives if len(normalize_text(n)) >= 1}
    return sorted(negatives)

def build_training_data(positive_terms: Sequence[str], seed: int = 42):
    negatives = _derive_pseudo_negatives(positive_terms, seed=seed)
    positives = list(dict.fromkeys([normalize_text(t) for t in positive_terms if len(normalize_text(t)) >= 2]))
    y_pos = [1] * len(positives)
    y_neg = [0] * len(negatives)

    X = positives + negatives
    y = y_pos + y_neg
    return X, y, set(positives)

def train_term_filter(csv_path: str, column_name: str = "Japanese", limit: int = 99999, seed: int = 42, extra_positive_terms: Sequence[str] = ()):
    terms = read_dictionary_terms(csv_path, column_name=column_name, limit=limit)
    if extra_positive_terms:
        terms = list(dict.fromkeys(terms + [normalize_text(t) for t in extra_positive_terms if len(normalize_text(t)) >= 2]))

    X, y, known_terms = build_training_data(terms, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)),
        ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = classification_report(y_test, preds, target_names=["non_term", "term"], output_dict=True)
    return model, known_terms, metrics

def discover_terms_in_text(text: str, model: Pipeline, known_terms: Set[str], threshold: float = 0.7, top_k: int = 80, use_ginza: bool = True) -> List[TermDiscoveryResult]:
    candidates = extract_candidates(text, use_ginza=use_ginza)
    if not candidates:
        return []

    probs = model.predict_proba(candidates)
    term_scores = probs[:, 1]

    results = []
    for cand, score in zip(candidates, term_scores):
        is_known = cand in known_terms
        if is_known or score >= threshold:
            results.append(
                TermDiscoveryResult(candidate=cand, score=float(score), is_known_dictionary_term=is_known)
            )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]