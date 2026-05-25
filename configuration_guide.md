# ⚙️ Hướng Dẫn Cấu Hình (Configuration Guide)

Tài liệu này giải thích chi tiết ý nghĩa các tham số cấu hình trong thư mục `configs/` và cung cấp hướng dẫn thiết lập tối ưu để đạt kết quả phát hiện bất thường tốt nhất.

---

## 1. 📂 Cấu Trúc File Cấu Hình YAML

Các file cấu hình YAML được chia làm 4 nhóm chính: `data` (dữ liệu), `model` (mô hình), `training` (huấn luyện), và `logging` (nhật ký).

Dưới đây là ví dụ cấu hình chuẩn cho mô hình lai Mamba-CNN:

```yaml
data:
  raw_dir: "data/raw/B04"
  processed_dir: "data/processed/B04"
  train_datasets: ["data/processed/B02", "data/processed/B05"]
  test_datasets: ["data/processed/B02", "data/processed/B03", "data/processed/B04", "data/processed/B05"]
  sampling_rate: 128000
  highpass_freq: 2000
  label_strategy: 'rms'
  window_stride: 1024
  lookback: 4096
  horizon: 1024
  skip_ratio: 0.05
  train_ratio: 0.4

model:
  patch_size: 16
  patch_stride: 8
  trend_downsample: 64
  cnn_out_channels: 64
  mamba_d_model: 64
  mamba_n_layer: 4 
  mamba_d_state: 16
  mamba_d_conv: 3
  mamba_expand: 3
  bidirectional: false
  decomp_kernel: 25 
  auto_scale_baselines: true
  use_decomposition: true
  use_stats: true

training:
  batch_size: 128
  learning_rate: 5e-4
  epochs: 10
  device: "cuda"
```

---

## 2. 🔍 Giải Thích Các Tham Số Cốt Lõi & Hướng Dẫn Thiết Lập Tối Ưu

### A. Nhóm Dữ Liệu (`data`)

- **`highpass_freq` (Mặc định: `2000` Hz)**: Tần số lọc thông cao cho tín hiệu rung thô.
  - *Ý nghĩa vật lý*: Giúp loại bỏ các thành phần nhiễu tần số thấp từ động cơ nền và làm nổi bật các xung va đập cơ học tần số cao do vết nứt vòng bi gây ra.
- **`lookback` (Ví dụ: `4096`) & `horizon` (Ví dụ: `1024`)**:
  - *Độ dài chuỗi lịch sử (`lookback`)*: Cần đủ dài để bao quát tối thiểu 2-3 chu kỳ quay của vòng bi. Một giá trị nhỏ như `512` thích hợp cho các thí nghiệm nhanh (config `snano.yaml`), nhưng giá trị từ `2048` đến `4096` mang lại khả năng nắm bắt thông tin chu kỳ dài hạn tốt nhất.
  - *Độ dài chuỗi dự báo (`horizon`)*: Chọn ở mức vừa phải (128 đến 1024). Dự báo quá dài sẽ làm giảm độ chính xác, dự báo quá ngắn sẽ không đủ thông tin để tính Anomaly Score ổn định.
- **`train_ratio` (Mặc định: `0.4` hoặc `0.5`)**: Tỷ lệ mẫu ở đầu chu kỳ sống dùng để huấn luyện.
  - *Quy tắc*: Chỉ huấn luyện trên giai đoạn hoạt động khỏe mạnh ban đầu (Healthy State). Tránh đặt `train_ratio` quá lớn (>0.6) vì có thể đưa tín hiệu suy thoái ban đầu vào tập huấn luyện, làm mô hình học cả trạng thái lỗi.
- **`skip_ratio` (Mặc định: `0.05`)**: Bỏ qua 5% dữ liệu ban đầu.
  - *Lý do*: Giai đoạn bắt đầu hoạt động (run-in period) thường có rung động không ổn định do thiết bị đang rà khớp, dễ gây nhiễu cho mô hình.

---

### B. Nhóm Mô Hình & Cơ Chế Thực Nghiệm Tối Ưu (`model`)

- **`auto_scale_baselines` (`true`)**: Tự động co giãn tham số baselines.
  - *Ý nghĩa*: Khi đặt là `true`, script huấn luyện sẽ tự động điều chỉnh số chiều ẩn (`hidden_dim`, `d_model`) của các mô hình đối chứng (LSTM, ModernTCN, PatchTST, SimpleMamba) sao cho tổng số lượng tham số học tập của chúng tương đương với mô hình lai Mamba-CNN (~200k - 300k tham số). Điều này đảm bảo sự **so sánh công bằng tuyệt đối** về dung lượng tính toán (Fair Parameter Budget).
- **Chuẩn hóa z-score & RevIN (Quyết định loại bỏ cốt lõi)**:
  - *Phân tích khoa học*: RevIN thực hiện chuẩn hóa z-score cục bộ trên từng cửa sổ dữ liệu. Khi vòng bi đi vào giai đoạn hỏng hóc nặng, biên độ tín hiệu rung (RMS) tăng vọt. Nếu bật RevIN hoặc chuẩn hóa z-score không kiểm soát, thuật toán sẽ co kéo biên độ cửa sổ lỗi về mức chuẩn hóa thông thường, làm mất đi sự khác biệt biên độ so với giai đoạn khỏe mạnh. Kết quả là sai số dự báo (Anomaly Score) không tăng ở giai đoạn cuối. **Bắt buộc phải tắt RevIN và chuẩn hóa z-score không phù hợp để mô hình nhạy bén với sự suy thoái biên độ.**
- **`use_stats: true` (Stats Head - Vật lý dẫn hướng)**: Kích hoạt đầu Stats Head.
  - *Ý nghĩa*: Stats Head trích xuất 8 đặc trưng thống kê miền thời gian (RMS, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Peak-to-Peak, Variance) từ cửa sổ lookback và đưa vào làm đặc trưng bổ trợ. Điều này giúp hướng dẫn mô hình bằng tri thức vật lý cơ học dòng máy, cải thiện đáng kể độ chính xác so với việc chỉ học chuỗi thời gian thuần túy.

---

### C. Nhóm Huấn Luyện (`training`)

- **`learning_rate` (Mặc định: `5e-4`)**: Tốc độ học tập.
  - *Lý do*: Phù hợp cho sự hội tụ ổn định của các lớp Mamba và CNN mà không bị bùng nổ gradient.
- **Hàm mất mát (`HuberLoss`)**: Sử dụng tổn thất Huber thay vì MSE thuần túy trong quá trình huấn luyện.
  - *Lý do*: Tín hiệu rung cơ học trong môi trường công nghiệp thường chứa các xung gai đột biến (outliers) do va đập ngẫu nhiên không phải lỗi. HuberLoss hoạt động như L1-loss với các lỗi lớn (ít bị ảnh hưởng bởi outliers) và hoạt động như L2-loss với các lỗi nhỏ (giúp hội tụ mịn).
