# CSE457 • LAB 3: MÔ HÌNH HMM VÀ NHẬN DẠNG TIẾNG NÓI TỪ ĐƠN

- **Sinh viên thực hiện:** Phạm Văn Giang — **MSSV:** 2351260649
- **Lớp:** Trí tuệ nhân tạo - Khóa 65 — **Khoa:** Công nghệ thông tin, Trường Đại học Thủy lợi
- **Môn học:** CSE457 - Xử lý âm thanh và tiếng nói

---

## 1. Giới thiệu tổng quan
Hệ thống nhận dạng từ đơn cô lập tiếng Việt sử dụng đặc trưng **MFCC 39 chiều động** ($\text{MFCC} + \Delta + \Delta^2$) kết hợp mô hình thống kê xác suất chuỗi ẩn **Left-to-Right Gaussian HMM**:
- **Tiền xử lý & Trích xuất đặc trưng:**
  * Lọc tiền nhấn $\alpha=0.97$, cắt tỉa khoảng lặng (trim silence) có bảo toàn trường độ tối thiểu.
  * 13 MFCC tĩnh (Hamming window 25ms, hop 10ms, 26 Mel filterbanks, $F_{\max}=8000\text{ Hz}$).
  * 13 Delta bậc 1 ($\Delta$) và 13 Delta bậc 2 ($\Delta^2$) mô tả vận tốc và gia tốc biến thiên phổ.
  * Chuẩn hóa trung bình và phương sai mức phát ngôn (Utterance-level CMVN) triệt tiêu méo kênh.
- **Mô hình âm học (Acoustic Model):**
  * Cấu trúc Left-to-Right (Bakis topology) 5 trạng thái ẩn với ràng buộc $a_{ii}=0.7, a_{i, i+1}=0.3, a_{N-1, N-1}=1.0$, cấm nhảy lùi/nhảy cóc.
  * Phân phối quan sát Gaussian đa biến với ma trận hiệp phương sai đường chéo (`covariance_type='diag'`).
  * Khởi tạo thông minh bằng phân đoạn thời gian đều (Uniform Temporal Segmentation).
  * Tối ưu hóa tham số bằng giải thuật Baum-Welch (EM).
- **Giải mã & Đánh giá (Decoding & Evaluation):**
  * Phân loại Maximum Likelihood: $\hat{w} = \arg\max_w \log P(X \mid \lambda_w)$.
  * Căn chỉnh trạng thái Viterbi (Viterbi State Alignment) ánh xạ frame sang các pha phát âm.
  * So sánh đối chứng chuẩn xác 1-1 với giải thuật DTW (Lab 2).
  * Tái xếp hạng (MAP Re-ranking) với mô hình ngôn ngữ Bigram làm mịn Add-$k$.

---

## 2. Cấu trúc thư mục

```text
TH3/
├── Lab 3.pdf                 # Đề bài thực hành chính thức của học phần
├── README.md                 # Tài liệu hướng dẫn sử dụng và báo cáo kỹ thuật
├── results.csv               # Bảng kết quả nhận dạng chi tiết từng file test
├── lexicon.txt               # Từ điển phát âm (Pronunciation Lexicon)
│
├── Lab03_2351260649.ipynb     # Jupyter Notebook chính thức (đầy đủ code, đồ thị, 10 câu hỏi)
├── Lab03.ipynb                # Bản sao đồng bộ của Notebook
│
├── feature_extractor.py      # Trích xuất đặc trưng 39-D MFCC + Deltas + CMVN
├── hmm_model.py              # Xây dựng, huấn luyện và giải mã Left-to-Right Gaussian HMM
├── dtw_baseline.py           # Bộ nhận dạng DTW (Lab 2) phục vụ so sánh đối chứng
├── language_model.py         # Bộ phân tích Lexicon, Bigram Add-k và MAP Re-ranking
├── plot_utils.py             # Công cụ vẽ biểu đồ, Confusion Matrix và Viterbi path (300 DPI)
├── experiments.py            # Kịch bản thực thi toàn bộ 6 thí nghiệm bắt buộc (E1 - E6)
├── prepare_data.py           # Phân chia dữ liệu train/test cho 1 speaker và 2 speakers
├── main.py                   # Chương trình CLI chạy toàn bộ pipeline từ đầu đến cuối
│
├── dataset/                  # Dữ liệu Speaker 1 (train: 01..03, test: 04..05)
├── dataset_speaker2/         # Dữ liệu Speaker 2 (dùng cho thí nghiệm Mismatch E5)
├── dataset_multi/            # Dữ liệu kết hợp cả 2 speaker (6 train + 4 test mỗi lớp)
├── models/                   # Thư mục lưu trữ 5 mô hình HMM đã huấn luyện (.pkl)
└── figures/                  # Thư mục chứa toàn bộ đồ thị thực nghiệm chất lượng cao
```

---

## 3. Hướng dẫn chạy chương trình

### 3.1. Cài đặt môi trường
Đảm bảo đã cài đặt Python $\ge 3.9$ và các gói phụ thuộc:
```bash
pip install numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn pandas
```

### 3.2. Chuẩn bị tập dữ liệu
Tự động phân chia dữ liệu thành cấu trúc chuẩn `train/` và `test/`:
```bash
python3 prepare_data.py
```

### 3.3. Chạy toàn bộ pipeline và thí nghiệm
Chạy chương trình chính để huấn luyện mô hình, kiểm thử, xuất đồ thị và báo cáo:
```bash
python3 main.py
```

### 3.4. Khởi chạy Jupyter Notebook
Mở và thực thi trực tiếp trên giao diện tương tác:
```bash
jupyter notebook Lab03_2351260649.ipynb
```

---

## 4. Kết quả Thực nghiệm Chính

### 4.1. Hiệu năng nhận dạng Baseline
* **Độ chính xác (Test Accuracy):** **100.00%** (10/10 file kiểm thử nhận dạng chính xác tuyệt đối).
* **Khoảng cách điểm (Score Margin):** Trung bình giữa Top-1 và Top-2 đạt **2469.3** log-likelihood units, không có hiện tượng phân vân hay nhập nhằng giữa các từ.

### 4.2. Bảng so sánh Đối chứng DTW (Lab 2) vs HMM (Lab 3)
Theo đúng yêu cầu mục 5.8 trong tài liệu Lab 3:

| Chỉ số đánh giá | DTW (Lab 2) | HMM (Lab 3) | Nhận xét phân tích |
| :--- | :---: | :---: | :--- |
| **Độ chính xác (Accuracy)** | 100.0% | 100.0% | Cả hai đều phân loại hoàn hảo trên tập từ đơn kiểm thử. |
| **Số mẫu train/lớp** | 3 mẫu | 3 mẫu | DTW lưu nguyên mẫu template; HMM học phân phối xác suất thống kê. |
| **Thời gian huấn luyện** | 0.0000s | ~0.040s | DTW không cần huấn luyện; HMM cần thời gian lặp kỳ vọng EM (Baum-Welch). |
| **Thời gian test / mẫu** | **239.22 ms** | **1.29 ms** | **HMM chạy nhanh hơn xấp xỉ 185 lần so với DTW.** |
| **Khả năng giải thích** | Warping Path | Viterbi State Path | HMM thể hiện trực quan các pha cấu âm (phụ âm $\to$ nguyên âm). |

---

## 5. Tóm tắt 6 Thí nghiệm Bắt buộc (E1 – E6)

1. **E1 (Số lượng state $n\_states \in \{3, 5, 7\}$):**
   * $n=3$: 90.0% (bị nhầm giữa các âm có cấu trúc phụ âm - nguyên âm phức tạp).
   * $n=5$: 100.0% (tối ưu nhất về số lượng tham số và cấu trúc âm tiết tiếng Việt).
   * $n=7$: 100.0% (đạt độ chính xác cao nhưng tốn thêm thời gian tính toán và nguy cơ overfit khi dữ liệu ít).
2. **E2 (Đặc trưng 13-D tĩnh vs 39-D động):**
   * Cả 2 đều đạt 100% trên tập test sạch; tuy nhiên đặc trưng 39-D tạo ra score margin giữa Top-1 và Top-2 vượt trội hơn 40%, tăng cường độ bền vững trước nhiễu.
3. **E3 (Hiệp phương sai Diagonal vs Full):**
   * Diagonal Covariance: 100.0% (train 0.04s).
   * Full Covariance: 80.0% (train 0.51s, tụt 20% do quá khớp / overfitting ma trận $39 \times 39$).
4. **E4 (So sánh benchmark DTW vs HMM):**
   * HMM chiếm ưu thế áp đảo về tốc độ suy luận trong môi trường thời gian thực (1.29 ms vs 239.22 ms).
5. **E5 (Speaker Mismatch):**
   * Huấn luyện trên Speaker 1 và kiểm thử trên Speaker 2 vẫn giữ vững độ chính xác nhờ việc chuẩn hóa CMVN và đặc trưng động.
6. **E6 (Trọng số ngôn ngữ $\lambda_{LM}$):**
   * Khi $\lambda_{LM}$ trong khoảng $0.1 - 0.3$, mô hình ngôn ngữ hỗ trợ tốt cho quyết định âm học. Khi $\lambda_{LM} \ge 1.0$, điểm ngôn ngữ bắt đầu lấn át bằng chứng âm học.
7. **Phân tích Ngưỡng Reject:**
   * Điểm trung bình đúng lớp: $-31.75$ / frame. Điểm trung bình sai lớp: $-70.88$ / frame.
   * Ngưỡng từ chối đề xuất: $\tau = -47.43$ / frame giúp phát hiện từ ngoài từ điển (OOV).

---

## 6. Đáp án 10 Câu hỏi Báo cáo (Mục 6.1)
*Được trình bày đầy đủ, chi tiết với công thức toán học và biểu đồ minh chứng trong file [`Lab03_2351260649.ipynb`](Lab03_2351260649.ipynb).*
