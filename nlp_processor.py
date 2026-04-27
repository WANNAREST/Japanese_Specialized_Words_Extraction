# nlp_processor.py
import re
import spacy
from typing import List

from config import JAPANESE_STOPWORDS
from text_utils import normalize_text, clean_ocr_text, clean_term_punctuation

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

def extract_candidates(text: str, min_len: int = 2, max_len: int = 15, use_ginza: bool = True) -> List[str]:
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