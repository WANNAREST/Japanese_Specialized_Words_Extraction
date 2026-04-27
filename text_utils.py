# text_utils.py
import re
import unicodedata

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text).strip())
    # Xóa các cụm trong ngoặc
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