# main.py
import json
from config import (
    DEFAULT_CSV_PATH, 
    DEFAULT_PDF_PATH, 
    PAGE_REPORT_DIR, 
    MERGED_CSV_PATH, 
    CONFIRMED_TERMS_PATH
)
from pdf_handler import run_pdf_discovery_all_pages
if __name__ == "__main__":
    print("Bắt đầu quá trình nạp từ điển và quét PDF...")
    metrics, hits, total_pages = run_pdf_discovery_all_pages(
        csv_path=DEFAULT_CSV_PATH,
        pdf_path=DEFAULT_PDF_PATH,
        page_report_dir=PAGE_REPORT_DIR,
        merged_csv_path=MERGED_CSV_PATH,
        confirmed_terms_path=CONFIRMED_TERMS_PATH,
        limit=99999,
        threshold=0.72,
        use_ginza=True,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Tổng số trang đã quét: {total_pages}")
    print(f"Tổng số thuật ngữ tìm thấy: {len(hits)}")
    print(f"File kết quả tổng hợp: {MERGED_CSV_PATH}")