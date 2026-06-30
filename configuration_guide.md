# ⚙️ Hướng Dẫn Cấu Hình (Configuration Guide)

Tài liệu này giải thích chi tiết ý nghĩa các tham số cấu hình trong thư mục `configs/` và cung cấp hướng dẫn thiết lập tối ưu để đạt kết quả phát hiện bất thường tốt nhất.

---

## 1. 📂 Cấu Trúc File Cấu Hình YAML

Các file cấu hình YAML được chia làm 4 nhóm chính: `data` (dữ liệu), `model` (mô hình), `training` (huấn luyện), và `logging` (nhật ký).

Dưới đây là ví dụ cấu hình chuẩn cho mô hình lai Mamba-CNN:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  train_datasets: ["data/processed/B02", "data/processed/B05", "data/processed/B08", "data/processed/B10", "data/processed/B11", "data/processed/B17"]
  test_datasets: ["data/processed/B01", "data/processed/B03", "data/processed/B04", "data/processed/B08", "data/processed/B10", "data/processed/B12", "data/processed/B17"]
  sampling_rate: 128000
  highpass_freq: 0          # TẮT bộ lọc thông cao — xử lý tín hiệu thô (paper không dùng filtering)
  label_strategy: 'rms'
  window_stride: 1024
  lookback: 4096            # L_x = 4096 — khớp Bảng 3 bài báo
  horizon: 512              # L_y = 512 (user override; paper dùng 1024)
  skip_ratio: 0.05
  train_ratio: 0.4

model:
  patch_size: 16
  patch_stride: 8
  trend_downsample: 64
  mamba_d_model: 64
  mamba_n_layer: 4 
  mamba_d_state: 16
  mamba_d_conv: 3
  mamba_expand: 4           # Hệ số mở rộng (mamba_expand)
  bidirectional: false      # Unidirectional — đúng với bài báo
  decomp_alpha: 0.1         # Khởi tạo alpha cho EMA Series Decomposition
  decomp_learnable: true    # Tự động học alpha qua backprop
  auto_scale_baselines: true
  use_decomposition: true   # BẬT Series Decomposition (Đóng góp cốt lõi #1)
  use_stats: true           # BẬT Physics-Informed Stats Head 8 chiều (Đóng góp cốt lõi #2)

training:
  batch_size: 64            # Kích thước batch
  learning_rate: 5e-4
  epochs: 10                # Số epoch huấn luyện (10 epochs theo bài báo)
  device: "cuda"
  loss_type: "huber"        # Huber loss δ=1.0
```

---

## 2. 🔍 Giải Thích Các Tham Số Cốt Lõi & Hướng Dẫn Thiết Lập Tối Ưu

### A. Nhóm Dữ Liệu (`data`)

- **`highpass_freq` (Mặc định: `0` Hz - TẮT)**: Tần số lọc thông cao cho tín hiệu rung thô.
  - *Ý nghĩa thực nghiệm*: Đặt về `0` để tắt bộ lọc thông cao và huấn luyện trực tiếp trên tín hiệu thô, giữ lại trọn vẹn thông tin biên độ cơ bản theo thiết kế tối ưu mới nhất của bài báo TSP.
- **`lookback` (Ví dụ: `4096`) & `horizon` (Ví dụ: `512`)**:
  - *Độ dài chuỗi lịch sử (`lookback`)*: Cần đủ dài để bao quát tối thiểu 2-3 chu kỳ quay của vòng bi. Giá trị `4096` khớp với Bảng 3 bài báo để mang lại khả năng nắm bắt thông tin chu kỳ dài hạn tốt nhất.
  - *Độ dài chuỗi dự báo (`horizon`)*: Đặt ở mức `512` (user override) thay vì `1024` để giảm thiểu độ trễ tính toán mà vẫn đảm bảo tính ổn định cao của Anomaly Score.
- **`train_ratio` (Mặc định: `0.4` hoặc `0.5`)**: Tỷ lệ mẫu ở đầu chu kỳ sống dùng để huấn luyện.
  - *Quy tắc*: Chỉ huấn luyện trên giai đoạn hoạt động khỏe mạnh ban đầu (Healthy State). Tránh đặt `train_ratio` quá lớn (>0.6) vì có thể đưa tín hiệu suy thoái ban đầu vào tập huấn luyện, làm mô hình học cả trạng thái lỗi.
- **`skip_ratio` (Mặc định: `0.05`)**: Bỏ qua 5% dữ liệu ban đầu.
  - *Lý do*: Giai đoạn bắt đầu hoạt động (run-in period) thường có rung động không ổn định do thiết bị đang rà khớp, dễ gây nhiễu cho mô hình.

---

### B. Nhóm Mô Hình & Cơ Chế Thực Nghiệm Tối Ưu (`model`)

- **`auto_scale_baselines` (`true`)**: Tự động co giãn tham số baselines.
  - *Ý nghĩa*: Khi đặt là `true`, script huấn luyện sẽ tự động điều chỉnh số chiều ẩn (`hidden_dim`, `d_model`) của các mô hình đối chứng (LSTM, PatchTST, SimpleMamba) sao cho tổng số lượng tham số học tập của chúng tương đương với mô hình lai Mamba-CNN (~200k - 300k tham số). Điều này đảm bảo sự **so sánh công bằng tuyệt đối** về dung lượng tính toán (Fair Parameter Budget).
- **`use_stats: true` (Stats Head - Vật lý dẫn hướng)**: Kích hoạt đầu Stats Head.
  - *Ý nghĩa*: Stats Head trích xuất 8 đặc trưng thống kê miền thời gian (RMS, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Peak-to-Peak, Variance) từ cửa sổ lookback và đưa vào làm đặc trưng bổ trợ. Điều này giúp hướng dẫn mô hình bằng tri thức vật lý cơ học dòng máy, cải thiện đáng kể độ chính xác so với việc chỉ học chuỗi thời gian thuần túy.

- **`decomp_alpha` (Mặc định: `0.1`) & `decomp_learnable` (Mặc định: `true`)**:
  - *Ý nghĩa*: Cấu hình cho bộ phân rã chuỗi thời gian bằng phương pháp đường trung bình lũy thừa (EMA Decomposition). Khi `decomp_learnable: true`, hệ số $\alpha$ khởi đầu bằng `decomp_alpha` sẽ tự động được học thông qua lan truyền ngược (backpropagation) để tìm ra hệ số làm mịn tối ưu $\alpha \approx 0.03$ thích hợp nhất với tín hiệu rung vòng bi thực tế.

---

### C. Nhóm Huấn Luyện (`training`)

- **`learning_rate` (Mặc định: `5e-4`)**: Tốc độ học tập.
  - *Lý do*: Phù hợp cho sự hội tụ ổn định của các lớp Mamba và CNN mà không bị bùng nổ gradient.
- **Hàm mất mát (`HuberLoss`)**: Sử dụng tổn thất Huber thay vì MSE thuần túy trong quá trình huấn luyện.
  - *Lý do*: Tín hiệu rung cơ học trong môi trường công nghiệp thường chứa các xung gai đột biến (outliers) do va đập ngẫu nhiên không phải lỗi. HuberLoss hoạt động như L1-loss với các lỗi lớn (ít bị ảnh hưởng bởi outliers) và hoạt động như L2-loss với các lỗi nhỏ (giúp hội tụ mịn).
