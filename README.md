
# Hệ thống Trích xuất Thuật ngữ Kỹ thuật Đường sắt Nhật Bản

Dự án này là một quy trình (Pipeline) tự động hóa hoàn toàn việc đọc tài liệu PDF chuyên ngành, làm sạch nhiễu OCR và sử dụng Trí tuệ nhân tạo (NLP + Machine Learning) để trích xuất các thuật ngữ kỹ thuật tinh khiết nhất. Hệ thống được tinh chỉnh đặc biệt để loại bỏ các câu văn diễn giải, khẩu lệnh lái tàu và các lỗi định dạng văn bản phức tạp.

## ✨ Tính năng cốt lõi

1.  **Xử lý ngôn ngữ sâu với GiNZA:** Sử dụng mô hình `ja_ginza` để phân tích ngữ pháp thực tế, nhận diện các danh từ ghép chuyên ngành.
2.  **Học máy chấm điểm (ML Scoring):** Sử dụng `Logistic Regression` để học cấu trúc từ vựng, giúp phân biệt giữa thuật ngữ kỹ thuật và ngôn ngữ đời thường.
3.  **Bộ lọc "Kỷ luật thép" (Strict Filtering):**
    * **Giới hạn độ dài:** Chỉ lấy các cụm từ từ 2 đến 15 ký tự (loại bỏ các câu văn dài).
    * **Chặn trợ từ:** Loại bỏ các cụm chứa `の` (của), `的な` (mang tính), giúp trích xuất thuật ngữ lõi thay vì cụm miêu tả.
    * **Dọn dẹp OCR:** Tự động sửa lỗi số thứ tự dính liền (vd: `1,運転台` -> `運転台`), nối lại chữ bị đứt đoạn do khoảng trắng.
    * **Chặn khẩu lệnh & Thông số:** Tự động loại bỏ các đơn vị đo lường (`kPa`, `度`) và khẩu lệnh xác nhận của tài xế (`よし`, `確認`).

## 📂 Cấu trúc dự án
Dự án được chia thành 6 mô-đun để dễ dàng quản lý:
* `config.py`: Cấu hình Stopwords và đường dẫn file.
* `text_utils.py`: Các hàm xử lý văn bản thô và dọn rác OCR.
* `nlp_processor.py`: Bộ lọc thuật ngữ và tích hợp GiNZA/Regex.
* `ml_engine.py`: Huấn luyện mô hình AI và dự đoán xác suất thuật ngữ.
* `pdf_handler.py`: Logic đọc file PDF, tìm ngữ cảnh (context) và xuất báo cáo.
* `main.py`: File thực thi chính kết nối toàn bộ quy trình.
## ⚙️ Hướng dẫn cài đặt
**1. Cài đặt các thư viện Python cần thiết:**
```bash
pip install pandas scikit-learn pymupdf spacy
```
**2. Tải mô hình ngôn ngữ GiNZA:**
```bash
python -m spacy download ja_ginza
```
## 🚀 Cách sử dụng

1.  **Chuẩn bị dữ liệu:** * Tạo thư mục `data_input/` và đặt file PDF tài liệu + file CSV từ điển vào đó.
    * Tạo thư mục `data_output/` để hệ thống xuất kết quả.
2.  **Cấu hình:** Mở file `config.py` để cập nhật tên file chính xác của bạn.
3.  **Chạy ứng dụng:**
```bash
python main.py
```
## 📊 Giải thích kết quả đầu ra

File `data_output/reports/terms_by_page.csv` với các cột:

| Cột | Ý nghĩa |
| :--- | :--- |
| **term** | Thuật ngữ đã được làm sạch hoàn toàn. |
| **page** | Số trang chứa thuật ngữ đó trong file PDF. |
| **context** | Nguyên văn đoạn văn chứa từ đó (dùng để đối chiếu). |
| **tag** | `KNOWN` (Đã có trong từ điển) hoặc `NEW` (Thuật ngữ mới phát hiện). |
| **score** | Độ tin cậy của AI (0.0 - 1.0). Càng cao càng chính xác. |

## 🛠 Tinh chỉnh (Fine-tuning)

* **Để lấy kết quả khắt khe hơn:** Tăng `threshold` trong `main.py` lên `0.8`.
* **Để lấy nhiều từ hơn:** Giảm `threshold` xuống `0.6`.
* **Thay đổi độ dài từ:** Chỉnh sửa tham số `max_len` trong `nlp_processor.py`.
