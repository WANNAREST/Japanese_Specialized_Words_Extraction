import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Set

import fitz
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


JAPANESE_STOPWORDS = {
    "こと", "もの", "ため", "とき", "ところ", "これ", "それ", "あれ", "どれ", "ここ", "そこ",
    "あそこ", "そして", "しかし", "また", "または", "及び", "また", "で", "に", "を", "が",
    "は", "へ", "と", "や", "の", "する", "した", "いる", "ある", "ない", "なる",
    "から", "まで", "より", "について", "として", "にて", "以外", "以上", "以下", "未満"
}


def load_ginza_model():
    try:
        return spacy.load(
            "ja_ginza",
            config={
                "nlp": {"tokenizer": {"split_mode": "C"}},
                "components": {"compound_splitter": {"split_mode": "C"}},
            },
        )
    except Exception:
        return None


GINZA_NLP = load_ginza_model()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text).strip())
    text = re.sub(r"[［\[\(].*?[］\]\)]", "", text).strip()
    return re.sub(r"\s+", " ", text)


def clean_term_punctuation(token: str) -> str:
    # 1. Dọn dẹp các dấu câu ở hai đầu thuật ngữ
    token = token.strip("「」『』【】（）()［］[]・.,、，。；:;： ")
    
    # 2. Xóa số mục lục bị dính ở đầu từ khóa (VD: "1,", "12.", "3、")
    token = re.sub(r"^\d{1,3}[.,、，．]\s*(?![系形号両線番])", "", token)
    
    # 3. Xóa tiếp số thứ tự nếu OCR để lại khoảng trắng (VD: "1 運転台")
    token = re.sub(r"^\d{1,3}\s+(?![系形号両線番])", "", token)
    
    # 4. Strip lại các dấu câu một lần cuối
    return token.strip("「」『』【】（）()［］[]・.,、，。；:;： ")


def clean_ocr_text(text: str) -> str:
    """Clean common OCR artifacts while preserving useful term content."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    
    # Phá hủy toàn bộ dấu ngoặc đơn/kép trước khi đưa cho GiNZA 
    text = re.sub(r"[「」『』【】（）()［］\[\]]", " ", text)
    
    # Biến dấu ・ (và) thành khoảng trắng để ép AI tách riêng 2 thuật ngữ
    text = text.replace("・", " ") 

    # Nối lại các chữ bị OCR tách rời bởi khoảng trắng
    text = re.sub(r"([ァ-ン])\s+([ァ-ン])", r"\1\2", text)  # Katakana (ハ ンドル -> ハンドル)
    text = re.sub(r"([一-龯])\s+([一-龯])", r"\1\2", text)  # Kanji (抑 速 -> 抑速)

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Dọn các số mục lục phức tạp
        line = re.sub(
            r"^\s*[・●○◯◎※\.]*\s*(?:\d{1,3}[\s\-]*)+[\.,、，．]?\s*(?![系形号両線番])(?=(?:[A-ZＡ-Ｚ]{2,}|[一-龯ぁ-んァ-ン]{2,}))",
            "",
            line,
        )

        # Remove inline ordinal numbers before obvious term tokens
        line = re.sub(
            r"(?:(?<=^)|(?<=[\s、。・,，;；:：]))(?:\d{1,3}[\s\-]*)+(?![系形号両線番])(?=(?:[A-ZＡ-Ｚ]{2,}|[一-龯ぁ-んァ-ン]{2,}))",
            "",
            line,
        )

        cleaned_lines.append(re.sub(r"\s+", " ", line).strip())

    return "\n".join(cleaned_lines)


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

    uniq_terms = list(dict.fromkeys(terms))
    return uniq_terms[:limit]


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
            if len(p) == 1:
                negatives.add(p)
            if 2 <= len(p) <= 3:
                negatives.add(p)
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


def train_term_filter(
    csv_path: str,
    column_name: str = "Japanese",
    limit: int = 99999,
    seed: int = 42,
    extra_positive_terms: Sequence[str] = (),
):
    terms = read_dictionary_terms(csv_path, column_name=column_name, limit=limit)
    if extra_positive_terms:
        terms = list(dict.fromkeys(terms + [normalize_text(t) for t in extra_positive_terms if len(normalize_text(t)) >= 2]))

    X, y, known_terms = build_training_data(terms, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = classification_report(y_test, preds, target_names=["non_term", "term"], output_dict=True)
    return model, known_terms, metrics


@dataclass
class TermDiscoveryResult:
    candidate: str
    score: float
    is_known_dictionary_term: bool


@dataclass
class TermHit:
    term: str
    page: int
    context: str
    tag: str
    score: float


def _is_term_like(token: str) -> bool:
    # 1. KỶ LUẬT THÉP VỀ ĐỘ DÀI
    if len(token) <= 1 or len(token) > 15:
        return False
        
    if re.fullmatch(r"\d+", token):
        return False

    # 2. CẤM "の", "的な", "であり" VÀ TỪ NỐI
    if "の" in token or "的な" in token:
        return False
    if re.search(r"(または|あるいは|および|かつ|ならびに|などは|などの|について|により|による|における|に対する|であり|及び|以外)", token):
        return False

    # 3. CẤM TÍNH TỪ, ĐỘNG TỪ MIÊU TẢ, DANH TỪ TRẠNG THÁI
    if re.search(r"(良い|適切|行う|となる|伴う|用いる|従う|応じる|省略|場合|必要|設定|投入)", token):
        return False

    # 4. CHẶN TRỢ TỪ NẰM Ở GIỮA CỤM TỪ
    if re.search(r"([でをがは]|に)(乗継|使用|設定|投入|繰返し|保安|抜|する|なる|よる)", token):
        return False

    # 5. CHẶN ĐUÔI CÂU, PHÓ TỪ
    if re.search(r"(ほか|へ|は次|より|から|まで|で|に|を|は|が|と|こと|受|押し|させ|ならな|ない|よし|とし|後|済|以上|以下|未満|する|した|している|される|て|れ|る)$", token):
        return False

    # 6. Chặn thông số đo lường
    if re.search(r"\d+(?:\.\d+)?\s*(kPa|kpa|Pa|V|A|W|kW|Hz|度|°|km|m|cm|mm|kg|t|秒|分|時間)", token, flags=re.IGNORECASE):
        return False

    # 7. Chặn cụm kết thúc bằng số nhưng chữ phía trước dài (đã bao gồm dấu ー)
    if re.search(r"[ァ-ン一-龯ぁ-んー]{4,}\d+$", token):
        return False

    # 8. SIẾT CHẶT TỶ LỆ CHỮ MỀM (HIRAGANA) XUỐNG CÒN 30%
    hira_count = len(re.findall(r"[ぁ-ん]", token))
    if hira_count / max(len(token), 1) > 0.3:
        return False

    # 9. Cấm token chứa khoảng trắng (Lỗi do OCR hoặc 2 từ ghép sai)
    if re.search(r"\s", token):
        return False

    # 10. Chặn Header/Footer
    if "年" in token and re.search(r"\d", token): 
        return False
    if re.search(r"(差\s*替|目\s*次|ページ)", token):
        return False
    if re.fullmatch(r"[\d\s・]+[一二三四五六七八九十]+", token):
        return False

    return True


def _extract_candidates_regex(text: str, min_len: int = 2, max_len: int = 15) -> List[str]:
    text = normalize_text(clean_ocr_text(text))
    raw = re.findall(r"[一-龯ぁ-んァ-ンーA-Za-z0-9・／/\-]{2,}", text)

    cands = []
    for token in raw:
        token = clean_term_punctuation(normalize_text(token))
        if not (min_len <= len(token) <= max_len):
            continue
        if token in JAPANESE_STOPWORDS:
            continue
        if not _is_term_like(token):
            continue
        cands.append(token)
    return list(dict.fromkeys(cands))


def _extract_candidates_ginza(text: str, min_len: int = 2, max_len: int = 15) -> List[str]:
    if GINZA_NLP is None:
        return []

    text = normalize_text(clean_ocr_text(text))
    if not text:
        return []

    doc = GINZA_NLP(text)
    cands = []

    for chunk in doc.noun_chunks:
        cand = clean_term_punctuation(normalize_text(chunk.text))
        if not (min_len <= len(cand) <= max_len):
            continue
        if cand in JAPANESE_STOPWORDS:
            continue
        if not _is_term_like(cand):
            continue
        cands.append(cand)

    for tok in doc:
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue
        cand = clean_term_punctuation(normalize_text(tok.text))
        if not (min_len <= len(cand) <= max_len):
            continue
        if cand in JAPANESE_STOPWORDS:
            continue
        if not _is_term_like(cand):
            continue
        cands.append(cand)

    return list(dict.fromkeys(cands))


def extract_candidates(
    text: str,
    min_len: int = 2,
    max_len: int = 15,
    use_ginza: bool = True,
) -> List[str]:
    ginza_cands = _extract_candidates_ginza(text, min_len=min_len, max_len=max_len) if use_ginza else []
    regex_cands = _extract_candidates_regex(text, min_len=min_len, max_len=max_len)

    merged = []
    seen = set()
    for cand in ginza_cands + regex_cands:
        if cand in seen:
            continue
        seen.add(cand)
        merged.append(cand)
    return merged


def discover_terms_in_text(
    text: str,
    model: Pipeline,
    known_terms: Set[str],
    threshold: float = 0.7,
    top_k: int = 80,
    use_ginza: bool = True,
) -> List[TermDiscoveryResult]:
    candidates = extract_candidates(text, use_ginza=use_ginza)
    if not candidates:
        return []

    probs = model.predict_proba(candidates)
    term_scores = probs[:, 1]

    results: List[TermDiscoveryResult] = []
    for cand, score in zip(candidates, term_scores):
        is_known = cand in known_terms
        if is_known or score >= threshold:
            results.append(
                TermDiscoveryResult(
                    candidate=cand,
                    score=float(score),
                    is_known_dictionary_term=is_known,
                )
            )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


def split_paragraphs(text: str) -> List[str]:
    if not text:
        return []

    text = clean_ocr_text(text)
    blocks = [normalize_text(b) for b in re.split(r"\n\s*\n", text) if b.strip()]
    if blocks:
        return blocks
    return [normalize_text(line) for line in text.split("\n") if line.strip()]


def find_context_for_term(paragraphs: Sequence[str], term: str) -> str:
    for p in paragraphs:
        if term in p:
            return p
    return ""


def save_discovery_report(results: Sequence[TermDiscoveryResult], report_path: str):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    payload = [
        {
            "candidate": r.candidate,
            "score": round(r.score, 6),
            "is_known_dictionary_term": r.is_known_dictionary_term,
            "type": "known" if r.is_known_dictionary_term else "new_candidate",
        }
        for r in results
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_terms_csv(term_hits: Sequence[TermHit], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = [
        {
            "term": h.term,
            "page": h.page,
            "context": h.context,
            "tag": h.tag,
            "score": round(h.score, 6),
        }
        for h in term_hits
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")


def run_mapping_visualize(pdf_path, terms_list, target_page, output_img_path):
    doc = fitz.open(pdf_path)
    page = doc[target_page - 1]

    found_in_page = []
    page_text = unicodedata.normalize("NFKC", page.get_text()).replace("\n", " ")

    for term in terms_list:
        if term in page_text:
            found_in_page.append(term)
            areas = page.search_for(term)
            for rect in areas:
                page.draw_rect(rect, color=(0, 0.8, 0), width=1.5)

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(output_img_path)
    doc.close()
    return found_in_page


def run_train_and_discover(
    csv_path: str,
    text: str,
    report_path: str = "data_output/reports/term_discovery.json",
    confirmed_terms_path: str = "data_output/reports/confirmed_new_terms.txt",
    column_name: str = "Japanese",
    limit: int = 99999,
    threshold: float = 0.72,
    use_ginza: bool = True,
):
    extra_terms = load_confirmed_terms(confirmed_terms_path)
    model, known_terms, metrics = train_term_filter(
        csv_path=csv_path,
        column_name=column_name,
        limit=limit,
        extra_positive_terms=extra_terms,
    )

    results = discover_terms_in_text(
        text=text,
        model=model,
        known_terms=known_terms,
        threshold=threshold,
        use_ginza=use_ginza,
    )
    save_discovery_report(results, report_path)
    return metrics, results


def run_pdf_discovery_all_pages(
    csv_path: str,
    pdf_path: str,
    page_report_dir: str = "data_output/reports/page_reports",
    merged_csv_path: str = "data_output/reports/terms_by_page.csv",
    confirmed_terms_path: str = "data_output/reports/confirmed_new_terms.txt",
    column_name: str = "Japanese",
    limit: int = 99999,
    threshold: float = 0.72,
    use_ginza: bool = True,
):
    extra_terms = load_confirmed_terms(confirmed_terms_path)
    model, known_terms, metrics = train_term_filter(
        csv_path=csv_path,
        column_name=column_name,
        limit=limit,
        extra_positive_terms=extra_terms,
    )

    os.makedirs(page_report_dir, exist_ok=True)
    all_hits: List[TermHit] = []

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_text_raw = doc[page_idx].get_text("text")
        page_text = normalize_text(page_text_raw)
        paragraphs = split_paragraphs(page_text_raw)

        results = discover_terms_in_text(
            text=page_text,
            model=model,
            known_terms=known_terms,
            threshold=threshold,
            top_k=500,
            use_ginza=use_ginza,
        )

        page_payload = []
        for r in results:
            tag = "KNOWN" if r.is_known_dictionary_term else "NEW"
            context = find_context_for_term(paragraphs, r.candidate)
            hit = TermHit(
                term=r.candidate,
                page=page_num,
                context=context,
                tag=tag,
                score=r.score,
            )
            all_hits.append(hit)

            page_payload.append(
                {
                    "candidate": r.candidate,
                    "score": round(r.score, 6),
                    "tag": tag,
                    "context": context,
                }
            )

        out_json = os.path.join(page_report_dir, f"page_{page_num:03d}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "page": page_num,
                    "hits": page_payload,
                    "count": len(page_payload),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    doc.close()

    save_terms_csv(all_hits, merged_csv_path)
    return metrics, all_hits, total_pages


if __name__ == "__main__":
    CSV = "data_input/鉄道技術用語辞典　第3版 1(Sheet1) (1).csv"
    PDF = "data_input/14 動力車乗務員作業標準（在来線）電車編（2024.4.1改正済） (1).pdf"

    metrics, hits, total_pages = run_pdf_discovery_all_pages(
        csv_path=CSV,
        pdf_path=PDF,
        page_report_dir="data_output/reports/page_reports",
        merged_csv_path="data_output/reports/terms_by_page.csv",
        confirmed_terms_path="data_output/reports/confirmed_new_terms.txt",
        limit=99999,
        threshold=0.72,
        use_ginza=True,
    )

    print("Model metrics (pseudo-label validation)")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Scanned pages: {total_pages}")
    print(f"Collected term hits: {len(hits)}")
    print("Per-page JSON reports: data_output/reports/page_reports")
    print("Merged CSV report: data_output/reports/terms_by_page.csv")