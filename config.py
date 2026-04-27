# Tập hợp các từ nối, hư từ tiếng Nhật cần loại bỏ
JAPANESE_STOPWORDS = {
    "こと", "もの", "ため", "とき", "ところ", "これ", "それ", "あれ", "どれ", "ここ", "そこ",
    "あそこ", "そして", "しかし", "また", "または", "及び", "また", "で", "に", "を", "が",
    "は", "へ", "と", "や", "の", "する", "した", "いる", "ある", "ない", "なる",
    "から", "まで", "より", "について", "として", "にて", "以外", "以上", "以下", "未満"
}

# Đường dẫn mặc định (Bạn hãy sửa lại cho đúng với máy của bạn)
DEFAULT_CSV_PATH = "data_input/鉄道技術用語辞典　第3版 1(Sheet1) (1).csv"
DEFAULT_PDF_PATH = "data_input/運転士関係マニュアル_1-08_動力車乗務員作業標準(在来線)異常時編(2025.09.18).pdf"
CONFIRMED_TERMS_PATH = "data_output/reports/confirmed_new_terms.txt"
PAGE_REPORT_DIR = "data_output/reports/page_reports"
MERGED_CSV_PATH = "data_output/reports/terms_by_page_運転士関係マニュアル_1-08_動力車乗務員作業標準(在来線)異常時編(2025.09.18).csv"