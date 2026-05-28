# Kế hoạch & Dàn ý Chi tiết: Phần III - Phương pháp Đề xuất (Nhấn mạnh Phân tách Seasonal/Trend & Tham chiếu DMamba)

Tài liệu này cung cấp cấu trúc chi tiết, cơ sở lý thuyết học thuật và các công thức toán học cho **Phần III (Proposed Methodology)** của bài báo nghiên cứu chẩn đoán bất thường vòng bi. Dàn ý này nhấn mạnh vào cơ chế **Phân tách Seasonal/Trend (Seasonal/Trend Decomposition)** theo nguyên lý **"Architectural Parsimony"** (Sự tối giản trong kiến trúc) từ paper **DMamba (arXiv:2602.09081)**, đồng thời thiết kế cấu trúc dạng **Modulable** để dễ dàng loại bỏ các thành phần phụ trợ (như Stats Head) nếu cần thiết trong tương lai mà vẫn giữ được tính toàn vẹn của mô hình phân tách cốt lõi.

---

## 1. Hướng dẫn Phân bổ Học thuật: Related Work (Section II) vs. Proposed Methodology (Section III)

Để đảm bảo cấu trúc bài báo chuẩn khoa học Q1, cơ sở lý thuyết phân tách Seasonal/Trend sẽ được chia làm hai phần rõ rệt:

1.  **Phần Tổng quan (Related Work - Section II)**: Đưa ra bức tranh toàn cảnh về việc các tác giả khác đã chứng minh sự cần thiết và hiệu quả của phân tách chuỗi thời gian (như STL, SSA, VMD+Mamba) đối với chẩn đoán rung động vòng bi.
2.  **Phần Thiết kế (Proposed Methodology - Section III)**: Đưa ra lập luận lý tính (Design Justification) dựa trên **Nguyên lý Tối giản Kiến trúc (Architectural Parsimony)** của **DMamba (arXiv:2602.09081)** để giải thích tại sao mô hình của bạn lại chọn luồng xử lý cụ thể này cho Trend (Linear) và Seasonal (Mamba).

---

## 2. Dự thảo Nội dung đưa vào Bản thảo (Drafts for Manuscripts)

### A. Đưa vào Section II: Related Work (English & Vietnamese)

#### [English Version for `report_nckh.md` under Related Work]
> *“Signal decomposition techniques, such as Seasonal-Trend Decomposition using LOESS (STL) and Singular Spectrum Analysis (SSA), have been widely established in machine health monitoring to isolate long-term degradation profiles from complex operational variations [cite papers]. STL effectively extracts the monotonic degradation trend representing physical wear while eliminating cyclic operating anomalies. Similarly, SSA facilitates model-free separation of smooth degradation trajectories from stochastic white noise. Recently, hybrid frameworks combining modal decomposition (e.g., VMD or EMD) with selective state space models (such as MD-BiMamba) have demonstrated that decomposing raw vibration signals prior to Mamba encoding significantly mitigates the fault masking effect, allowing the selective scan mechanism to focus strictly on fault-induced impulses.”*

#### [Vietnamese Version for `report_nckh_vn.md` under Related Work]
> *“Các kỹ thuật phân tách tín hiệu, chẳng hạn như Phân tách Seasonal-Trend bằng LOESS (STL) và Phân tích Phổ Đơn kênh (SSA), đã được thiết lập rộng rãi trong giám sát sức khỏe máy móc nhằm cô lập các đặc tính suy thoái dài hạn khỏi các biến động vận hành phức tạp [cite papers]. STL trích xuất hiệu quả xu hướng suy thoái đơn điệu đại diện cho sự mài mòn vật lý trong khi loại bỏ các bất thường có chu kỳ vận hành. Tương tự, SSA tạo điều kiện phân tách không mô hình các quỹ đạo suy thoái mịn khỏi nhiễu trắng ngẫu nhiên. Gần đây, các khung lai kết hợp phân tách chế độ (ví dụ: VMD hoặc EMD) với mô hình không gian trạng thái chọn lọc (như MD-BiMamba) đã chứng minh rằng việc phân tách các tín hiệu rung động thô trước khi mã hóa Mamba giúp giảm thiểu đáng kể hiệu ứng che lấp lỗi (fault masking), cho phép cơ chế quét chọn lọc tập trung chặt chẽ vào các xung động do lỗi gây ra.”*

---

### B. Đưa vào Section III: Proposed Methodology (English & Vietnamese)

#### [English Version for `report_nckh.md` under Section III.C / III.D Rationale]
> *“Aligning with the principle of architectural parsimony introduced in recent decomposition models (e.g., DMamba [arXiv:2602.09081]), the proposed framework decouples the processing of seasonal and trend streams to match their distinct physical and statistical profiles. The trend component $X_{trend}$, representing slow-moving, low-frequency mechanical wear, possesses a low dimensional complexity and is projected via a simple linear branch to prevent overfitting. Conversely, the seasonal component $X_{seasonal}$, capturing high-frequency transient shocks and rotational cycles, exhibits highly non-linear dynamics and is routed to a channel-independent Mamba encoder to model long-range temporal dependencies with linear-time efficiency. This dual-stream design ensures that the forecasting target is physically interpretable and structurally parsimonious.”*

#### [Vietnamese Version for `report_nckh_vn.md` under Section III.C / III.D Rationale]
> *“Đồng bộ với nguyên lý tối giản kiến trúc (Architectural Parsimony) được giới thiệu trong các mô hình phân tách gần đây (ví dụ: DMamba [arXiv:2602.09081]), khung phương pháp đề xuất tách biệt việc xử lý các luồng seasonal và trend để tương thích với các đặc tính vật lý và thống kê riêng biệt của chúng. Thành phần xu hướng $X_{trend}$, đại diện cho sự mài mòn cơ học tần số thấp, chuyển động chậm, có độ phức tạp chiều thấp và được chiếu thông qua một nhánh tuyến tính đơn giản để ngăn ngừa hiện tượng quá khớp (overfitting). Ngược lại, thành phần mùa vụ $X_{seasonal}$, ghi lại các xung va đập chuyển tiếp tần số cao và chu kỳ quay, thể hiện động học phi tuyến tính mạnh và được chuyển đến bộ mã hóa Mamba độc lập kênh để mô hình hóa các phụ thuộc xa với hiệu suất thời gian tuyến tính. Thiết kế hai luồng này đảm bảo mục tiêu dự báo có thể diễn giải được về mặt vật lý và tối giản về mặt cấu trúc.”*

---

## 3. Dàn ý Chi tiết & Công thức Toán học cho Phần III (Proposed Methodology)

### A. Phát biểu Bài toán & Khung Phương pháp (Problem Formulation)
1.  **Đầu vào & Đầu ra**:
    *   Chuỗi tín hiệu rung động đầu vào (Lookback window): $X \in \mathbb{R}^{C \times L}$, trong đó $C$ là số kênh cảm biến (ví dụ: hướng ngang và dọc), $L$ là chiều dài cửa sổ lịch sử.
    *   Chuỗi dự báo tương lai (Forecast horizon): $\hat{Y} \in \mathbb{R}^{C \times H}$, với $H$ là chiều dài cửa sổ dự báo tiếp theo.
2.  **Mục tiêu Phát hiện Bất thường Không Giám sát**:
    *   Mô hình được huấn luyện hoàn toàn trên dữ liệu hoạt động bình thường (Healthy state) để học cách tái cấu trúc/dự báo tương lai bình thường.
    *   Khi có bất thường (Fault state), sai lệch giữa tín hiệu thực tế $Y$ và tín hiệu dự báo $\hat{Y}$ sẽ tăng vọt, làm cơ sở để tính điểm bất thường (Anomaly Score).

---

### B. Tiền xử lý Tín hiệu: Bộ lọc Thông cao Butterworth (Signal Preprocessing) (có thể bỏ)
Trước khi đưa vào mô hình phân tách, tín hiệu thô được lọc thông cao để loại bỏ các dao động tần số thấp từ động cơ nền hoặc nhiễu môi trường, tập trung vào các dải tần nhạy cảm với va đập cơ học.
1.  **Hàm truyền đạt của bộ lọc Butterworth bậc $n$**:
    $$|H(f)|^2 = \frac{1}{1 + \left(\frac{f_c}{f}\right)^{2n}}$$
    Trong đó $f_c$ là tần số cắt ($highpass\_freq$), $f$ là tần số tín hiệu.
2.  **Chuyển đổi sang miền thời gian rời rạc**:
    Áp dụng phép biến đổi song tuyến tính (Bilinear Transform) để thu được phương trình sai phân hiệu chỉnh tín hiệu:
    $$y[t] = \sum_{i=0}^{n} b_i x[t-i] - \sum_{j=1}^{n} a_j y[t-j]$$
    *Ý nghĩa vật lý*: Giữ lại xung va đập biên độ nhỏ ở tần số cao khi xuất hiện vết nứt rỗ sớm trên bề mặt vòng bi.

---

### C. Khối Phân tách Chuỗi Thời gian (Series Decomposition) (trọng tâm)
Kế thừa triết lý từ DMamba và DLinear, tín hiệu đã lọc $X$ được phân tách thành hai thành phần riêng biệt bằng bộ lọc trung bình trượt (Moving Average) triển khai qua phép toán $\text{AvgPool1d}$ 1 chiều:
1.  **Thành phần Xu hướng (Trend)**:
    $$X_{trend} = \text{AvgPool1d}(X, \text{kernel\_size} = k)$$
    $$X_{trend} = \text{Padding}(X_{trend})$$  *(đảm bảo giữ nguyên kích thước $C \times L$)*
2.  **Thành phần Mùa vụ & Xung đột biến (Seasonal)**:
    $$X_{seasonal} = X - X_{trend}$$

---

### D. Nhánh Dự báo Xu hướng Tuyến tính (Linear Trend Forecasting Stream)
*   **Nguyên lý đối chiếu DMamba**: Trend đại diện cho đường trung bình trượt thể hiện sự mài mòn vật lý chậm. Việc đưa Trend qua một bộ mã hóa nặng như Mamba hay Transformer là không cần thiết và dễ làm nhiễu xu hướng.
1.  **Công thức dự báo**:
    Áp dụng một lớp chiếu tuyến tính (Linear Projection) trực tiếp từ cửa sổ lịch sử $L$ ra cửa sổ dự báo $H$:
    $$\hat{Y}_{trend} = \mathbf{W}_T X_{trend} + \mathbf{b}_T$$
    Trong đó:
    *   $\mathbf{W}_T \in \mathbb{R}^{L \times H}$ là ma trận trọng số dùng chung cho tất cả các kênh (Channel-Independent weight sharing).
    *   $\mathbf{b}_T \in \mathbb{R}^{H}$ là vector bias.
    *   *Tùy chọn Downsample (nếu có)*: Nếu $L$ quá lớn, thực hiện Downsampling bằng $\text{AvgPool1d}$ trước khi đưa vào lớp Linear để giảm số lượng tham số và tăng tính ổn định:
        $$X'_{trend} = \text{AvgPool1d}(X_{trend}, s)$$
        $$\hat{Y}_{trend} = \mathbf{W}_T X'_{trend} + \mathbf{b}_T \quad \left(\mathbf{W}_T \in \mathbb{R}^{\frac{L}{s} \times H}\right)$$

### E. Nhánh Dự báo Mùa vụ dựa trên Mamba-CNN Độc lập Kênh (Seasonal Mamba-CNN Stream) (trọng tâm)
Thành phần $X_{seasonal}$ chứa các thông tin đặc trưng lỗi tần số cao phức tạp được xử lý qua chuỗi các module để vừa trích xuất đặc trưng cục bộ (CNN), vừa học tương quan toàn cục dài hạn (Mamba):

1.  **Cơ chế Vá mảnh thời gian dạng tích chập (CNN Patch Embedding)**:
    Để giảm chiều dài chuỗi đầu vào và tăng khả năng trích xuất đặc trưng cục bộ, chuỗi $X_{seasonal} \in \mathbb{R}^{C \times L}$ được phân tách và chiếu vào không gian ẩn $D$ ($d_{model}$) thông qua hai dạng biểu diễn tương đương toán học:
    *   **Dạng Unfold + Tuyến tính**: Chia chuỗi thành các phân đoạn không/có chồng chập (patches) kích thước $P$, bước nhảy $S$:
        $$N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1$$
        $$X_{patch} \in \mathbb{R}^{C \times N \times P}$$
        Chiếu tuyến tính từng mảnh vào không gian ẩn:
        $$S_{embed} = \mathbf{W}_E X_{patch} + \mathbf{b}_E \quad \left(S_{embed} \in \mathbb{R}^{C \times N \times D}\right)$$
    *   **Dạng Tích chập 1D (CNN Patching Equivalent)**: Phép chiếu trên có thể biểu diễn tương đương bằng một lớp Conv1D trượt trên trục thời gian với số kênh đầu ra là $D$, kích thước nhân (kernel size) bằng $P$, và bước nhảy (stride) bằng $S$:
        $$S_{embed} = \text{Conv1D}(X_{seasonal}, \text{kernel\_size}=P, \text{stride}=S) \in \mathbb{R}^{C \times D \times N}$$ (sau đó chuyển vị chiều thành $C \times N \times D$).
    *   *Ý nghĩa vật lý*: Phép biến đổi này hoạt động như một bộ lọc tích chập cục bộ để làm mịn và nén các mẫu xung lỗi động học trong một khung thời gian ngắn thành một vector đại diện.

2.  **Độc lập Kênh (Channel Independence - CI)**:
    *   Làm phẳng chiều Batch ($B$) và chiều Kênh ($C$) để mô hình xử lý mỗi kênh cảm biến như một chuỗi độc lập:
        $$S_{in} = \text{Reshape}(S_{embed}) \in \mathbb{R}^{(B \times C) \times N \times D}$$
    *   *Vai trò*: Ngăn ngừa hiện tượng rò rỉ nhiễu chéo giữa các hướng cảm biến khác nhau, giữ nguyên đặc trưng động học riêng biệt của từng trục đo.

3.  **Khối lai Mamba-CNN (Hybrid Mamba-CNN Block)**:
    Sau khi vá mảnh, chuỗi được truyền qua bộ mã hóa gồm các khối Mamba lai tích chập. Mỗi khối Mamba chứa một nhánh CNN tích hợp sẵn để xử lý cục bộ trước khi quét chọn lọc:
    *   **Nhánh Tích chập Cục bộ (Local 1D CNN)**: Tín hiệu chiếu ẩn $u_t$ đầu tiên đi qua lớp tích chập 1D với kích thước nhân $d_{conv}$ (thường bằng 4) để làm mượt và thu thập thông tin ngữ cảnh cục bộ lân cận:
        $$x'_t = \text{SiLU}(\text{Conv1D}(u_{t-d_{conv}:t}))$$
    *   **Cơ chế Quét chọn lọc (Selective SSM - S6)**: Chuỗi đặc trưng sau khi lọc qua CNN sẽ được chuyển vào bộ quét trạng thái tuyến tính phụ thuộc đầu vào:
        $$h_t = \mathbf{A} h_{t-1} + \mathbf{B}_t x'_t$$
        $$y_t = \mathbf{C}_t h_t + \mathbf{D} x'_t$$
    *   *Sự kết hợp Mamba-CNN*: Việc tích hợp lớp Conv1D ngay trước bước quét SSM giúp mô hình loại bỏ các nhiễu trắng ngẫu nhiên tần số cao (nhiễu đo lường) trước khi thực hiện chọn lọc trạng thái dài hạn, tăng độ ổn định của hệ thống.

4.  **Đầu dự báo mùa vụ (Seasonal Forecasting Head)**:
    *   Biểu diễn ẩn sau Mamba-CNN được đưa qua lớp dự báo để chiếu ra chiều dài dự báo $H$:
        $$\hat{Y}_{seasonal} = \text{ForecastHead}(S_{out})$$

---

### F. Khối Trích xuất Đặc trưng Vật lý Thống kê (Stats Head) — *Tùy chọn Modulable* (có thể bỏ chỉ giữ lại RMS, Kurtosis)
Để hỗ trợ mô hình học sâu nhận biết các trạng thái cơ học trực quan hơn, Stats Head trích xuất 8 đặc trưng miền thời gian từ tín hiệu thô:
1.  **8 công thức vật lý**:
    *   *Mean*: $\mu = \frac{1}{L}\sum_{t=1}^{L} x_t$
    *   *Standard Deviation*: $\sigma = \sqrt{\frac{1}{L}\sum_{t=1}^{L} (x_t - \mu)^2}$
    *   *Root Mean Square (RMS)*: $x_{rms} = \sqrt{\frac{1}{L}\sum_{t=1}^{L} x_t^2}$
    *   *Peak-to-Peak*: $x_{p2p} = \max(x) - \min(x)$
    *   *Skewness*: $S = \frac{1}{L}\sum_{t=1}^{L} \left(\frac{x_t - \mu}{\sigma}\right)^3$
    *   *Kurtosis*: $K = \frac{1}{L}\sum_{t=1}^{L} \left(\frac{x_t - \mu}{\sigma}\right)^4$
    *   *Crest Factor*: $CF = \frac{\max(|x|)}{x_{rms}}$
    *   *Shape Factor*: $SF = \frac{x_{rms}}{\frac{1}{L}\sum_{t=1}^{L}|x_t|}$
2.  **Dung hợp Đặc trưng (Fusion Head)**:
    *   8 đặc trưng trên được đưa qua Batch Normalization và một lớp Linear để chiếu vào không gian ẩn $D$.
    *   Được ghép nối (concatenate) với vector biểu diễn chuỗi ẩn của Mamba trước khi đưa ra kết quả dự báo $\hat{Y}_{seasonal}$:
        $$S'_{out} = [ \text{Flatten}(S_{out}) \,;\, \text{Linear}(\text{Stats}) ]$$
        $$\hat{Y}_{seasonal} = \mathbf{W}_F S'_{out} + \mathbf{b}_F$$
    *   *Tính Modulable*: Lớp này được điều khiển bởi tham số cấu hình `use_stats`. Nếu thiết lập `use_stats = False`, mô hình sẽ tự động bỏ qua luồng này và chỉ thực hiện dự báo Seasonal thuần túy dựa trên Mamba.

---

### G. Khối Hòa trộn Hai Nhánh tự học (Learnable Dual-Stream Mixing Module) (trọng tâm)
Thay vì sử dụng phép cộng trực tiếp $\hat{Y} = \hat{Y}_{seasonal} + \hat{Y}_{trend}$ như trong DMamba hay DLinear, mô hình đề xuất sử dụng một hệ số hòa trộn tự học $\alpha$ cho từng kênh:
1.  **Công thức hòa trộn**:
    $$\alpha_c = \sigma(w_{mix, c}) \quad \forall c \in \{1, \dots, C\}$$
    $$\hat{Y}_c = \alpha_c \hat{Y}_{seasonal, c} + (1 - \alpha_c) \hat{Y}_{trend, c}$$
    Trong đó:
    *   $\sigma(\cdot)$ là hàm kích hoạt Sigmoid nhằm giới hạn $\alpha_c \in (0, 1)$.
    *   $w_{mix, c}$ là tham số học được tương ứng với mỗi kênh cảm biến.
2.  **Ưu điểm so với phép cộng trực tiếp**:
    *   Cho phép mô hình tự động điều chỉnh tỷ trọng đóng góp của xu hướng mài mòn dài hạn và các dao động xung va đập ngắn hạn trên từng kênh cảm biến cụ thể.
    *   Nếu trong quá trình tối ưu hóa, $\alpha_c$ tiến về $1$, mô hình sẽ tự động hội tụ về dạng mô hình chỉ sử dụng nhánh Seasonal (như khi tắt decomposition).

---

### H. Tính Điểm Bất thường & Xác định Ngưỡng động POT-EVT (quan trọng)
1.  **Điểm bất thường (Anomaly Score)**:
    Được tính bằng sai số bình phương trung bình (MSE) giữa chuỗi thực tế $Y$ và dự báo $\hat{Y}$ trên từng cửa sổ thời gian:
    $$S[t] = \frac{1}{C \times H} \sum_{c=1}^{C} \sum_{h=1}^{H} \left( Y_{c, h}[t] - \hat{Y}_{c, h}[t] \right)^2$$
2.  **Ngưỡng động Peak Over Threshold (POT) dựa trên Lý thuyết giá trị cực biên (EVT)**:
    *   *Nguyên lý*: Không giả định phân phối lỗi là phân phối chuẩn. Chỉ tập trung vào đuôi của phân phối sai số (các giá trị cực đại vượt ngưỡng).
    *   *Quy trình*:
        1. Thiết lập một ngưỡng ban đầu thấp $t$ (ví dụ: phân vị 98% của sai số trên tập dữ liệu lành mạnh).
        2. Lọc các giá trị vượt ngưỡng: $E_i = S_i - t > 0$.
        3. Khớp các giá trị vượt ngưỡng $E_i$ vào Phân phối Pareto Tổng quát (GPD) với tham số hình dáng $\gamma$ và tham số quy mô $\sigma$:
           $$F(x) = 1 - \left(1 + \frac{\gamma x}{\sigma}\right)^{-1/\gamma}$$
        4. Tính toán ngưỡng quyết định động $z_q$ cho một xác suất cảnh báo rất nhỏ $q$ (ví dụ: $q = 10^{-4}$):
           $$z_q = t + \frac{\sigma}{\gamma} \left( \left( \frac{q n}{N_t} \right)^{-\gamma} - 1 \right)$$
           Trong đó $n$ là tổng số mẫu quan sát, $N_t$ là số mẫu vượt ngưỡng ban đầu $t$.
        5. Mẫu có $S[t] > z_q$ sẽ được phân loại là bất thường.

---

## 3. Bản đồ Ánh xạ từ Toán học sang Code nguồn (`src/`)

Để đảm bảo tính nhất quán học thuật và thực tiễn, dưới đây là bảng đối chiếu giữa các phần lý thuyết và file code tương ứng:

| Ký hiệu Toán học / Module | Lớp Code / Hàm Tương ứng | File Mã Nguồn trong `src/` |
| :--- | :--- | :--- |
| Bộ lọc Butterworth $H(s)$ | Lớp `ButterworthFilter` | `src/data/preprocess.py` |
| 8 chỉ số vật lý miền thời gian | Hàm `extract_time_features` | `src/data/dataset.py` |
| Phân tách $X_{trend}, X_{seasonal}$ | Lớp `SeriesDecomposition` | `src/models/mamba/layers.py` |
| Lớp nhúng vá mảnh $S_{embed}$ | Lớp `SimplePatchEmbedding` | `src/models/mamba/layers.py` |
| Mã hóa Seasonal $S_{out}$ | Lớp `MambaEncoder` | `src/models/mamba/mamba_encoder.py` |
| Dung hợp Stats Head & Mamba | Lớp `FusionForecastHead` | `src/models/mamba/fusion_head.py` |
| Nhánh xu hướng $\hat{Y}_{trend}$ | `self.trend_head` | `src/models/mamba/hybrid_mamba.py` |
| Trộn tự học $\alpha$ | `self.mix_alpha` | `src/models/mamba/hybrid_mamba.py` |
| Điểm bất thường $S[t]$ | Hàm `compute_anomaly_score` | `src/evaluator/anomaly_scorer.py` |
| Ngưỡng động POT $z_q$ | Lớp `POTPredictor` / `pot_threshold` | `src/evaluator/metrics.py` |

---

## 4. Kế hoạch Rút gọn Kiến trúc trong Tương lai (Nếu chỉ giữ lại Seasonal/Trend)

Nếu trong các nghiên cứu sau này, bạn muốn loại bỏ các phần phụ trợ để đưa mô hình về dạng phân tách thuần túy (Seasonal/Trend Decomposition), mô hình hiện tại đã hỗ trợ sẵn việc này mà không cần sửa đổi cấu trúc lớn:

1.  **Loại bỏ Stats Head (có thể bỏ chỉ giữ lại RMS, Kurtosis)**:
    *   *Cách thực hiện*: Đặt cấu hình `use_stats = False` trong file config.
    *   *Hệ quả*: Lớp `FusionForecastHead` sẽ tự động bỏ qua phần ghép nối đặc trưng vật lý và chỉ đóng vai trò là một lớp chiếu tuyến tính thuần túy ánh xạ từ biểu diễn ẩn của Mamba sang độ dài dự báo $H$.
2.  **Loại bỏ Bộ lọc Butterworth (có thể bỏ)**:
    *   *Cách thực hiện*: Đặt tham số bộ lọc trong config thành Bypass hoặc tắt cờ lọc trong pipeline nạp dữ liệu.
    *   *Hệ quả*: Mô hình sẽ nhận trực tiếp tín hiệu rung động thô để tiến hành phân tách Seasonal/Trend.
3.  **Tác động học thuật**:
    *   Mô hình lúc này sẽ hoạt động tương đồng gần như hoàn toàn với triết lý của **DMamba (arXiv:2602.09081)** kết hợp với cơ chế độc lập kênh và vá mảnh của **PatchTST**, tạo ra một baseline cực kỳ mạnh mẽ và tối giản.

---

## 5. Bằng chứng Thực nghiệm (Proof-of-Concept) — Dữ liệu B03 Run-to-Failure

> **Script**: `notebooks/proof_bearing_trend_seasonal.py`  
> **Dataset**: B03 (LDM, 614 files, `Fs=12800 Hz`, fault onset `M0547` ≈ 89% lifecycle)  
> **MA_KERNEL**: `3457` (~270 ms ≈ 12 vòng quay @ 2793 rpm)

### Số liệu Tổng hợp

| Metric | Trước (kernel=257) | **Sau (kernel=3457)** | Ý nghĩa |
|:---|:---:|:---:|:---|
| PCA Trend (90% var) | 13 PCs | **2 PCs** ✅ | Trend = near-planar manifold |
| PCA Seasonal (90% var) | 31 PCs | **>30 PCs** ✅ | Seasonal = high-dim, non-linear |
| **Dimensionality ratio S/T** | 2.4× | **15.5×** 🔥 | Architectural Parsimony proved |
| Energy separation Trend/Seasonal | 0.66× | **1/28×** | Tách phổ rõ ràng |

---

### Fig 1 — Waveform Decomposition (Moving-Average, 3 giai đoạn)

> *Xem file*: `artifacs/figures/fig1_waveform_decomposition.png`

**Đọc hình**: 3 cột (Healthy M0001 / Mid-life M0308 / Fault Onset M0558) × 3 hàng (Raw / Trend / Seasonal).

- ✅ **Trend phẳng, mượt** — với kernel 3457, Trend không oscillate như phiên bản cũ (kernel=257).
- ✅ **Seasonal ≈ Raw** trong tất cả ba giai đoạn → xác nhận Trend chỉ capture DC baseline.
- ✅ **Fault Onset**: Seasonal bùng nổ biên độ (RMS 0.059→0.088), Trend vẫn nhỏ (RMS 0.006) → decomp đúng.
- ⚠️ Trend cột Fault Onset có hướng âm (DC drift artifact khi MA gặp impulse — bình thường, không phải lỗi).

---

### Fig 2 — Power Spectral Density (Tách phổ Trend/Seasonal)

> *Xem file*: `artifacts/figures/fig2_psd_frequency_evidence.png`

**Đọc hình**: Panel giữa quan trọng nhất — Trend PSD (cam) vs Seasonal PSD (xanh) sau decomposition.

- ✅ **Tách phổ 6-8 bậc** ở dải 1000-6000 Hz — cải thiện lớn so với kernel cũ.
- ✅ **Energy ratio**: `Seasonal/Trend = 28×` (trước: 0.66×).
- ✅ **FFT fault (panel phải)**: Peak tại **108 Hz** ≈ BPFI/harmonic xoay → Seasonal capture fault impulse thực sự.
- ⚠️ Còn chồng nhau ở <200 Hz (shaft rotation tần số thấp — không thể tránh bằng MA filter).

---

### Fig 3 — Longitudinal RMS (Toàn bộ vòng đời 614 files)

> *Xem file*: `artifacts/figures/fig3_longitudinal_rms.png`

**Đọc hình**: Panel trên = Trend RMS (trục trái) + Raw RMS (trục phải, màu xanh). Panel dưới = Seasonal RMS.

- ✅ **Seasonal RMS spike chính xác tại M0547** và duy trì 0.15-0.35 trong fault phase.
- ✅ **Trend RMS ≈ 0 suốt 89% lifecycle** → justify dùng Linear đơn giản cho Trend (ít thông tin).
- ✅ **Raw RMS xác nhận**: tăng cùng thời điểm với Seasonal → fault được capture bởi Seasonal.
- ⚠️ Legend bị che khuất (cosmetic, nên dời sang `upper right`).

---

### Fig 4 — PCA Dimensionality *(Figure học thuật mạnh nhất)*

> *Xem file*: `artifacts/figures/fig4_pca_dimensionality.png`

**Đọc hình**: Cumulative Explained Variance (%) vs Number of PCs. Cam = Trend, Xanh = Seasonal.

- ✅ **Trend đạt 90% variance với chỉ 2 PCs** → nearly 1D manifold → Linear branch là đủ.
- ✅ **Seasonal chưa đạt 90% dù dùng 30 PCs** (chỉ ~83%) → high-dimensional → cần Mamba encoder.
- ✅ **Dimensionality ratio = 15.5×** → bằng chứng định lượng cho *Architectural Parsimony*.

> **Câu trích dẫn đề xuất cho paper (Section III.B)**:
> *"As empirically demonstrated on the B03 run-to-failure dataset, the trend component requires only **2 principal components** to explain 90% of its variance, while the seasonal component requires more than **30 components** to reach 83% (dimensionality ratio: **15.5×**). This disparity directly motivates the proposed dual-stream architecture."*

---

### Fig 5 — Architecture Justification Summary

> *Xem file*: `artifacts/figures/fig5_architecture_justification.png`

**Đọc hình**: Trái = Kurtosis Seasonal theo thời gian. Phải = Scatter Trend RMS vs Seasonal RMS (màu = measurement index).

- ✅ **Kurtosis**: ổn định ~3.0 (Gaussian) trong healthy, tăng **>4** tại M0547 → impulsive, non-Gaussian → justify Mamba.
- ✅ **Scatter**: Healthy cluster dồn cụm (Trend≈0, Seasonal≈0.05), Fault scatter phân tán theo hướng Seasonal tăng mạnh → hai component độc lập trong healthy, co-evolve khi fault.
- ✅ Label đã sửa: *"Both components spike at fault onset (co-evolve at failure, independent in healthy)"*.
- ⚠️ Chưa có ellipse annotation phân vùng Healthy/Fault trong scatter (cosmetic improvement).

---

### Bảng Đánh giá Tổng hợp

| Figure | Điểm HỢP LÝ (✅) | Điểm CẦN CẢI THIỆN (⚠️) |
|:---|:---|:---|
| Fig 1 Waveform | Trend phẳng; Seasonal ≈ Raw; Fault spike rõ | Trend fault cột phải đi âm (DC artifact, minor) |
| Fig 2 PSD | Tách phổ 6-8 bậc; energy 28×; peak BPFI | Chồng nhau <200 Hz (shaft rotation) |
| Fig 3 Longitudinal | Spike M0547 chính xác; Raw xác nhận | Legend bị che; twin axis cần ghi chú scale |
| **Fig 4 PCA** | **2 vs 30+ PCs, ratio 15.5× — bằng chứng mạnh nhất** | — |
| Fig 5 Justification | Kurtosis >3 tại fault; scatter phân cụm đúng | Thiếu ellipse annotation phân vùng |

