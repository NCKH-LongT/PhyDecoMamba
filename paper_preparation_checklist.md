# 📋 Checklist Chuẩn Bị Viết Bài Báo Khoa Học (Paper Preparation Checklist)

Tài liệu này cung cấp danh sách kiểm tra chi tiết các đầu việc kỹ thuật và các quy tắc hành văn khoa học cần tuân thủ để phục vụ cho việc viết bài báo nghiên cứu đạt chuẩn Q1.

---

## 1. ⚙️ Trạng Thái Git & Đồng Bộ Mã Nguồn (Git & Source Code Verification)

Đảm bảo mã nguồn huấn luyện, kiểm thử và đánh giá đã được tách biệt hoàn toàn từ Jupyter Notebooks và được quản lý chặt chẽ trong Git.

- [x] **Mô-đun hóa mã nguồn**: Kiểm tra xem toàn bộ các thành phần của hệ thống đã được phân tách từ các file Notebook vào thư mục `src/` chưa.
  - [x] **Xử lý dữ liệu (`src/data/`)**: Đã có `MultiBearingDataset` hỗ trợ lọc thông cao (Butterworth high-pass filter), tính toán đặc trưng vật lý 8 chiều (Stats Head) và chia tập dữ liệu huấn luyện/kiểm thử theo thời gian.
  - [x] **Kiến trúc mô hình (`src/models/`)**: Đã chứa mô hình đề xuất `PhyDecoMamba` (được kế thừa tương thích ngược qua alias `HybridMambaCNN`) và các baselines so sánh (`LSTM`, `ModernTCN`, `PatchTST`, `SimpleMamba`).
  - [x] **Huấn luyện (`src/training/`)**: Đã chứa `train.py` thực hiện vòng lặp huấn luyện với cơ chế tự động co giãn tham số (auto-scaling) và dừng sớm (`EarlyStopping`).
  - [x] **Đánh giá (`src/evaluation/`)**: Đã chứa `eval.py` thực hiện đánh giá đa mô hình trên tập kiểm thử vòng bi độc lập và các giải thuật tính ngưỡng tự học (`metrics.py`, `anomaly_scorer.py`).
- [x] **Tích hợp các số liệu đo đạc thực nghiệm nâng cao**:
  - [x] **VRAM Tiêu Thụ Đỉnh (Peak VRAM)**: Được đo bằng `reset_peak_memory_stats` và `max_memory_allocated` sau mỗi đợt đánh giá mô hình.
  - [x] **Phân rã độ trễ 4 bước (Real-time latency profiling)**: Đo thời gian chi tiết của 4 bước: *Data Transfer* (đẩy dữ liệu lên GPU), *Model Inference* (chạy forward), *Anomaly Scoring* (tính MSE) và *Decision* (so sánh ngưỡng).
  - [x] **Độ trễ hiệu chuẩn ngưỡng (Calibration Overhead)**: Đo thời gian chạy hiệu chuẩn ngưỡng (GMM, Robust, POT...) tính bằng mili-giây trên tập dữ liệu khỏe mạnh của từng vòng bi.

---

## 2. 📝 Các Tài Liệu Hướng Dẫn Kèm Theo (Documentation Deliverables)

Tạo và hoàn thiện 3 tài liệu hướng dẫn kỹ thuật cốt lõi tại thư mục gốc của dự án:

- [x] **`running_guide.md` (Hướng dẫn chạy chương trình)**: Hướng dẫn chi tiết từ bước cài đặt môi trường CUDA, đến việc chạy huấn luyện và đánh giá mô hình bằng dòng lệnh.
- [x] **`configuration_guide.md` (Hướng dẫn cấu hình)**: Hướng dẫn chi tiết ý nghĩa các tham số trong file cấu hình YAML, cấu trúc Stats Head, và thiết lập các thí nghiệm.
- [x] **`source_code_description.md` (Mô tả mã nguồn)**: Mô tả kiến trúc thư mục dự án, luồng dữ liệu của hệ thống và giải thích chức năng của từng lớp/mô-đun trong mã nguồn.

---

## 3. ✍️ Quy Tắc Viết Bài Báo Khoa Học (Paper Writing Guidelines & Constraints)

Khi bắt tay vào viết bản thảo bài báo (Manuscript), cần tuân thủ nghiêm ngặt các quy định học thuật sau:

- [ ] **Giới hạn độ dài**: Bài báo chỉ viết ngắn gọn, súc tích trong phạm vi **15 - 18 trang** (bao gồm cả tài liệu tham khảo và hình vẽ). Tập trung tối đa vào đóng góp học thuật và kết quả thực nghiệm thay vì viết lan man.
- [ ] **Sử dụng thể bị động (Passive Voice)**: Trình bày toàn bộ nội dung nghiên cứu bằng thể bị động để đảm bảo tính khách quan khoa học.
- [ ] **Tuyệt đối không sử dụng đại từ ngôi thứ nhất/chủ động**:
  - **CẤM** các cụm từ như: *We propose* (Chúng tôi đề xuất), *Our model* (Mô hình của chúng tôi), *Our system* (Hệ thống của chúng tôi), *We evaluated* (Chúng tôi đã đánh giá), *In this paper, we...*
  - **Bảng quy đổi ví dụ cụ thể để tránh lỗi**:
    
    | ❌ Kiểu viết bị cấm (Active Voice) | ✅ Kiểu viết chuẩn khoa học (Passive Voice) |
    | :--- | :--- |
    | **We propose** a hybrid Mamba-CNN architecture... | **A hybrid Mamba-CNN architecture is proposed**... |
    | **Our model** outperforms the baselines... | **The proposed model outperforms** the baselines... |
    | **Our system** consists of three stages... | **The system is composed of** three stages... |
    | **We evaluate** the models on the B02 dataset... | **The models are evaluated** on the B02 dataset... |
    | **We calculate** the anomaly score using MSE... | **The anomaly score is computed** via Mean Squared Error (MSE)... |
    | **We choose** a lookback window of 4096... | **A lookback window of 4096 is selected**... |
    | **We train** the model for 10 epochs... | **The model is trained** for 10 epochs... |

---

## 4. 🔍 Kiểm Tra Tính Đúng Đắn Khoa Học (Scientific Rigor & Correctness)

Trước khi viết phần Phương pháp và Thực nghiệm, hãy kiểm tra và xác nhận các luận điểm khoa học sau để phản biện tốt với Reviewers Q1:

- [ ] **Stats Head (Vật lý dẫn đường - Physics-Informed)**:
  - Mô hình nhận đầu vào là chuỗi thời gian thô và được bổ trợ bởi **8 đặc trưng thống kê vật lý** được tính trực tiếp từ cửa sổ lookback trong `_compute_physical_stats`:
    1. **Mean** — giá trị trung bình tín hiệu
    2. **Std** — độ lệch chuẩn
    3. **RMS** — Root Mean Square (chỉ thị năng lượng rung)
    4. **Peak-to-Peak** — biên độ đỉnh-đỉnh (max − min)
    5. **Skewness** — độ lệch phân phối (bất đối xứng xung va đập)
    6. **Kurtosis** — độ nhọn phân phối (nhạy cảm với xung gai ngắn của vết nứt)
    7. **Crest Factor** — tỷ số đỉnh / RMS (chỉ thị giai đoạn đầu lỗi)
    8. **Shape Factor** — tỷ số RMS / Mean Absolute (đặc trưng hình dạng sóng)
  - Cấu trúc này hướng mô hình tập trung vào sự biến đổi cơ học thực tế thay vì chỉ học đặc trưng số học thuần túy, đảm bảo khả năng diễn giải vật lý (Physical Interpretability).
- [ ] **Hiệu chuẩn ngưỡng không rò rỉ dữ liệu (Leakage-Free Calibration)**:
  - Ngưỡng phát hiện bất thường (POT, 3-Sigma, Robust) được tính toán cục bộ cho từng vòng bi (Per-bearing calibration) dựa **chỉ** trên dữ liệu khỏe mạnh ở giai đoạn đầu (cửa sổ trong khoảng `skip_ratio` đến `train_ratio`, với nhãn `bearing_labels == 0`). Không có thông tin nào từ tập kiểm thử lỗi được sử dụng trong quá trình hiệu chuẩn, phản ánh chính xác quy trình triển khai công nghiệp thực tế.
- [ ] **So sánh công bằng tham số (Fair Parameter Budget)**:
  - Khi `auto_scale_baselines: true`, các mô hình đối chứng (LSTM, ModernTCN, PatchTST, SimpleMamba) được tự động điều chỉnh chiều ẩn để tổng số tham số huấn luyện được tương đương với mô hình đề xuất `PhyDecoMamba` (đại diện bởi `HybridMambaCNN`). Điều này đảm bảo sự so sánh công bằng thuần túy về hiệu năng chứ không phải về dung lượng mô hình.

