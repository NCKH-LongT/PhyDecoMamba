# 🚀 Hướng Dẫn Chạy Chương Trình (Running Guide)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, chuẩn bị dữ liệu, chạy huấn luyện (training) và chạy đánh giá (evaluation) cho hệ thống phát hiện bất thường dựa trên mô hình lai Mamba-CNN.

---

## 1. 📋 Yêu Cầu Hệ Thống & Cài Đặt (Prerequisites & Installation)

Mô hình Mamba yêu cầu môi trường tính toán có hỗ trợ GPU NVIDIA (CUDA) để đạt hiệu năng tối ưu và tránh lỗi biên dịch thư viện `mamba-ssm`.

### Yêu cầu phần cứng khuyến nghị:
- GPU NVIDIA kiến trúc Ampere trở lên (Compute Capability SM 8.0+, ví dụ: RTX 30xx/40xx, A100, H100).
- Hệ điều hành: Linux hoặc Windows (thông qua WSL2 hoặc cài đặt môi trường C++ thích hợp).

### Các bước cài đặt:

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Environment)**:
   ```bash
   python -m venv venv
   # Trên Windows:
   .\venv\Scripts\activate
   # Trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt các thư viện lõi**:
   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
   ```

3. **Cài đặt thư viện `mamba-ssm`**:
   ```bash
   pip install mamba-ssm --no-build-isolation
   ```

4. **Cài đặt các gói phụ thuộc khác**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. 💾 Chuẩn Bị Dữ Liệu (Dataset Preparation)

1. Tải bộ dữ liệu vòng bi PRONOSTIA hoặc Paderborn (B02/B04/B05).
2. Định dạng dữ liệu:
   - Dữ liệu sau khi trích xuất cần được chuyển đổi sang định dạng `.pt` (PyTorch Tensors) bằng cách sử dụng các script tiền xử lý tương ứng trong `src/data/pipeline.py` hoặc `src/data/preprocess_pronostia.py`.
   - Cấu trúc thư mục dữ liệu sau tiền xử lý:
     ```text
     data/processed/B04/
     ├── operating_conditions.csv
     ├── file_rms.json
     ├── data_B04_M0001.pt
     └── ...
     ```

> **⚠️ Quan trọng về chuẩn hóa tín hiệu**: Tín hiệu rung được lưu ở dạng **biên độ vật lý thô** (raw amplitude), **không áp dụng chuẩn hóa z-score hay instance normalization**. Đây là yêu cầu thiết kế bắt buộc: biên độ RMS tăng dần theo thời gian chính là chỉ thị suy thoái cốt lõi. Nếu chuẩn hóa từng cửa sổ, thông tin suy thoái dài hạn sẽ bị triệt tiêu và mô hình sẽ không thể phát hiện bất thường chính xác.

---

## 3. 🏋️ Huấn Luyện Mô Hình (Model Training)

Sử dụng script `src/training/train.py` để thực hiện huấn luyện mô hình Mamba và các baselines.

### Cú pháp chạy huấn luyện:
```bash
python src/training/train.py --config configs/default.yaml --model Mamba1-Hybrid
```

### Các tham số dòng lệnh quan trọng:
- `--config`: Đường dẫn tới file cấu hình YAML (ví dụ: `configs/default.yaml`, `configs/snano.yaml`).
- `--model`: Tên mô hình cần huấn luyện (`Mamba1-Hybrid` - định danh trong mã nguồn của mô hình đề xuất **PhyDecoMamba**, `SimpleMamba`, `LSTM`, `ModernTCN`, `PatchTST` hoặc `all` để huấn luyện tất cả tuần tự).
- `--epochs`: Ghi đè số lượng epoch huấn luyện từ cấu hình YAML (ví dụ: `--epochs 10`).
- `--batch_size`: Ghi đè kích thước batch (ví dụ: `--batch_size 128`).
- `--file_subset_ratio`: Tỷ lệ lấy mẫu tệp dữ liệu để tăng tốc độ huấn luyện mà vẫn giữ tính liên tục thời gian (Ví dụ: `--file_subset_ratio 5` sẽ lấy mẫu cứ mỗi 5 file dữ liệu).

*Sau khi kết thúc huấn luyện, checkpoint tốt nhất sẽ được lưu tự động tại `results/models/[model_slug]_[config_name]_best.pth` nhờ cơ chế Early Stopping.*

---

## 4. 📊 Đánh Giá Mô Hình (Model Evaluation)

Sử dụng script `src/training/eval.py` để chạy kiểm thử mô hình trên các vòng bi khác nhau và tính toán các chỉ số phát hiện bất thường.

### Cách 1: Đánh giá một mô hình đơn lẻ (Single Model Evaluation)
Đánh giá hiệu năng của một file checkpoint cụ thể:
```bash
python src/training/eval.py --config configs/default.yaml --model_type Mamba1-Hybrid --model_path results/models/mamba1_hybrid_default_best.pth
```

### Cách 2: Đánh giá so sánh đa mô hình (Multi-Model Comparison)
Tự động quét thư mục lưu trữ checkpoints để đánh giá so sánh hiệu năng dự báo và phát hiện bất thường của tất cả các mô hình:
```bash
python src/training/eval.py --config configs/default.yaml --models LSTM,ModernTCN,PatchTST,SimpleMamba,Mamba1-Hybrid --models_dir results/models
```

**Đầu ra kết quả bao gồm:**
- Bảng so sánh sai số dự báo (Forecasting Metrics): MAE, MSE, RMSE, MAPE.
- Bảng so sánh phát hiện bất thường (Anomaly Detection Metrics) ứng với các ngưỡng: 3-Sigma, Robust, POT (Peak-Over-Threshold), Self-Learn (GMM), và Optimal.
- Lượng tiêu thụ bộ nhớ đỉnh (Peak VRAM MB) của từng mô hình.
- Độ trễ thời gian thực chi tiết (Data Transfer, Inference, Anomaly Scoring, Decision Latency) tính bằng ms/sample.
- Độ trễ hiệu chuẩn (Calibration Overhead) cho từng thuật toán ngưỡng.

---

## 5. 📈 Trực Quan Hóa (Visualization)

Hệ thống cung cấp các công cụ trực quan hóa sau:

1. **Vẽ đồ thị xu hướng lỗi toàn bộ vòng đời (Full Life-cycle Trend)**:
   ```bash
   python src/evaluation/visualize_full_lifecycle.py --model_path results/models/mamba1_hybrid_default_best.pth --model_type Mamba1-Hybrid --config configs/default.yaml
   ```
2. **Kiểm tra lát cắt một file dữ liệu cụ thể (Single File Visualization)**:
   ```bash
   python src/evaluation/visualize_file.py --file data_B02_M1100.pt --model_path results/models/mamba1_hybrid_default_best.pth --model_type Mamba1-Hybrid --config configs/default.yaml
   ```
3. **Trực quan hóa xu hướng suy giảm trên tập test**:
   ```bash
   python src/evaluation/visualize_trend.py --model_path results/models/mamba1_hybrid_default_best.pth --model_type Mamba1-Hybrid --config configs/default.yaml
   ```
