# English Version

# Implementation Plan: Draft Expansion for Bearing Anomaly Detection Paper (With Detailed Outline)

This plan details the expansion of the research paper draft (`report_nckh.md` and `report_nckh_vn.md`) based on the current production-ready codebase in `src/`. It includes sanitising the Abstract/Introduction (removing RevIN and multi-scale patching) and outlining the structural content for Section II (Literature Review) and Section III (Proposed Methodology).

## User Review Required

> [!IMPORTANT]
> **Architecture Realignment**: All references to RevIN, Z-score normalization, PatchLSTM, and multi-scale patching were removed from the codebase to operate purely on raw signals and physical statistics. The existing draft of the paper (`report_nckh.md`) still lists RevIN and multi-scale series decomposition/patching as core contributions.
>
> We will update the Abstract and Introduction to match the actual codebase:
> *   **Remove**: Reversible Instance Normalization (RevIN).
> *   **Remove**: Multi-scale patching/decomposition.
> *   **Add**: Raw signals + 8 time-domain physical features (Stats Head) to guide the model with physical interpretability.
> *   **Add**: Hybrid Mamba-CNN architecture featuring Series Decomposition (seasonal/trend split) and a Fusion Forecasting Head.

## Proposed Changes

We will modify both the English (`report_nckh.md`) and Vietnamese (`report_nckh_vn.md`) drafts.

### Section I: Abstract & Introduction Sanitisation
*   Update Title (if necessary) to remove "Multi-Scale" and focus on Hybrid Mamba-CNN with physical statistics.
*   Rewrite the Abstract to reflect the removal of RevIN/multi-scale patching and highlight the Stats Head (8 physical statistics) and the Hybrid Mamba-CNN forecasting framework.
*   Rewrite the Introduction (especially the research questions and core contributions) to align with this raw signal + stats head + hybrid Mamba forecasting paradigm.

### Section II: Literature Review (New Section)
*   **Deep Learning for Sequence Modeling in PHM**: The limitations of standard deep learning sequence models for long-range bearing signals:
    *   *LSTM*: Sequential bottleneck and vanishing gradients.
    *   *TCN*: Receptive field size limitation under highly high-frequency signals.
    *   *Transformers*: Quadratic complexity ($O(N^2)$) and lack of local translation invariance making them highly sensitive to industrial background noise.
*   **Selective State Space Models (Mamba) for Time Series**: Why selective state space models are a superior fit for long-horizon mechanical degradation tracking due to their linear-time complexity and selective scan mechanism.
*   **Dynamic Thresholding Methods**: Limitations of rigid, static thresholds under variable operating conditions. Overview of dynamic thresholding techniques (3-Sigma, Robust/IQR, Gaussian Mixture Models, and Peak Over Threshold based on Extreme Value Theory).
*   **Literature Matrix Table**: Comparison of relevant SOTA papers from `references.md` (e.g., FEMamba, TFG-Mamba, ModernTCN, Autoformer).
*   **Experimental Baselines**: Formal introduction of baseline models for benchmarking, including:
    *   *LSTM*: Standard recurrent neural network for temporal dependency modeling.
    *   *TCN*: Temporal Convolutional Network with dilated convolutions.
    *   *ModernTCN*: Modernized convolutional time-series model.
    *   *PatchTST*: Patch-based channel-independent Transformer.
    *   *SimpleMamba*: Standard selective state space model without CNN hybrid or Stats Head.

### Section III: Proposed Methodology (New Section)
*   **Problem Formulation**: Formal definitions of the windowed input $X$, lookback window $L$, forecast horizon $H$, and reconstructed forecast $\hat{Y}$.
*   **Data Preprocessing**: Butterworth high-pass filtering ($highpass\_freq$) to isolate high-frequency mechanical impacts and remove motor noise.
*   **Physics-Informed Stats Head**: Mathematical equations for the 8 physical statistics extracted in `dataset.py` (Mean, Std, RMS, Peak-to-Peak, Skewness, Kurtosis, Crest Factor, Shape Factor) and how they are normalized and fused in `FusionForecastHead`.
*   **Hybrid Mamba-CNN Architecture**:
    *   *Series Decomposition*: $\text{AvgPool1d}$ filtering to split seasonal ($X_S$) and trend ($X_T$) components.
    *   *Trend Branch*: Downsampling and linear forecasting.
    *   *Seasonal Branch*: Linear patching ($SimplePatchEmbedding$), channel-independent Mamba encoder ($MambaEncoder$), and fusion forecasting ($FusionForecastHead$).
    *   *Learnable Mixing*: Mixing coefficient $\alpha$ controlled by a sigmoid parameter.
*   **Anomaly Scoring & Decision Boundary**:
    *   MSE reconstruction error calculation in `anomaly_scorer.py`.
    *   Extreme Value Theory (EVT) / Peak Over Threshold (POT) dynamic thresholding in `metrics.py` (using Generalized Pareto Distribution).
    *   Alternative calibration heads (3-Sigma, Robust/IQR, GMM).

---

## Detailed Outline of Sections (English)

### Section II: Literature Review
*   **A. Sequence Modeling Paradigms in Machine Prognostics**
    *   *Recurrent Architectures (LSTM)*: Review of recursive state updates; discussion of computational limits in parallel training and performance degradation on high-frequency signals due to vanishing gradients.
    *   *Convolutional Networks (TCN)*: Analysis of parallel computation; discussion of limitations imposed by fixed receptive fields when processing long degradation trajectories.
    *   *Attention-Based Models (Transformers)*: Critique of quadratic computation/memory complexity ($O(N^2)$); explanation of susceptibility to high-frequency industrial noise due to lack of local spatial-temporal bias.
*   **B. Selective State Space Models for Long-Sequence Forecasting**
    *   Introduction to continuous state space equations ($\mathbf{A}, \mathbf{B}, \mathbf{C}$ representations).
    *   The selective scan mechanism (Mamba) as a linear complexity sequence modeling framework.
    *   Analysis of current applications of Mamba in bearing remaining useful life (RUL) estimation and identification of the research gap: lack of unsupervised reconstruction-based anomaly detection models.
*   **C. Anomaly Thresholding Techniques in Prognostics**
    *   Critique of static thresholds under dynamic load/speed fluctuations.
    *   Review of dynamic/adaptive threshold methods: statistical (3-Sigma, IQR), clustering-based (Gaussian Mixture Models), and extreme value modeling (Peak Over Threshold).
*   **D. Scholarly Positioning Matrix**
    *   A literature matrix table comparing SOTA time-series models on: input type (raw/processed), architecture type, thresholding method, computational complexity, and main limitations.
*   **E. Baseline Benchmarking Models**
    *   LSTM, TCN, ModernTCN, PatchTST (Vanilla Transformer representation), and SimpleMamba implementations.
    *   Rationale for selecting these baseline models representing recurrent, convolutional, attention-based, and state-space paradigms.

### Section III: Proposed Methodology
*   **A. Framework Overview and Problem Formulation**
    *   Definition of input sequence $X \in \mathbb{R}^{C \times L}$ and predicted sequence $\hat{Y} \in \mathbb{R}^{C \times H}$.
    *   Formulation of the unsupervised anomaly detection paradigm via next-window forecasting discrepancy.
*   **B. Signal Preprocessing & Highpass Filtering**
    *   Butterworth filter formulation: Transfer function $H(s)$ and discrete implementation.
    *   Physical rationale for Butterworth filtering: blocking operational low-frequency noise (e.g., motor rotation) while preserving structural high-frequency mechanical shock impacts.
*   **C. Physics-Informed Stats Head**
    *   Mathematical equations and mechanical significance for:
        1. *Mean*: Central signal baseline.
        2. *Standard Deviation*: Energy variation.
        3. *Root Mean Square (RMS)*: Total mechanical energy / vibration severity indicator.
        4. *Peak-to-Peak*: Absolute maximum displacement amplitude.
        5. *Skewness*: Signal asymmetry (indicating impact directionality).
        6. *Kurtosis*: Transient shock indicator (highly sensitive to early pitting/spalling cracks).
        7. *Crest Factor*: Ratio of peak to RMS (indicator of early shock-to-energy ratio).
        8. *Shape Factor*: Indicator of wave shape changes.
    *   Batch Normalization and linear projection layer to map the 8 stats to the forecasting dimension.
*   **D. Multi-Scale Temporal Series Decomposition**
    *   Formulation of moving average decomposition: $X = X_{seasonal} + X_{trend}$.
    *   Trend extraction via 1-D Average Pooling: $X_{trend} = \text{AvgPool1d}(X, \text{kernel\_size})$.
    *   Linear projection of trend to forecasting length: $\hat{Y}_{trend} = \mathbf{W}_T X_{trend} + \mathbf{b}_T$.
*   **E. Seasonal Channel-Independent Mamba Encoder**
    *   *Patching Mechanism*: Linear projection of non-overlapping window segments to hidden dimension $d_{model}$ using `SimplePatchEmbedding`.
    *   *Channel Independence*: Flattening batch and channel dimensions to process multi-channel signals independently, preventing cross-channel noise leakage.
    *   *Mamba Encoder*: Selective State Space layer updates and sequence processing.
    *   *Fusion Head*: Combining the pooled Mamba sequence representation with the normalized 8-dimensional Stats Head features via concatenation and projecting to the forecasting window.
*   **F. Learnable Mixing Module**
    *   Formulation of learnable alpha mixing:
        $$\alpha = \sigma(\mathbf{w}_{mix})$$
        $$\hat{Y} = \alpha \odot \hat{Y}_{seasonal} + (1 - \alpha) \odot \hat{Y}_{trend}$$
    *   Discussion of how the mixing parameter dynamically balances low-frequency degradation trends and high-frequency seasonal shock transients.
*   **G. Forecasting-Based Anomaly Scoring and Dynamic POT Thresholding**
    *   Anomaly score formulation: Mean Squared Error (MSE) between actual future horizon $Y$ and predicted forecast $\hat{Y}$.
    *   Extreme Value Theory (EVT) / Peak Over Threshold (POT) threshold formulation:
        *   Setting initial high quantile threshold $t$.
        *   Fitting excess values $X - t$ to Generalized Pareto Distribution (GPD).
        *   Calculating threshold $z_q$ for target probability $q$:
            $$z_q = t + \frac{\sigma}{\gamma} \left( \left( \frac{q n}{N_t} \right)^{-\gamma} - 1 \right)$$
    *   Description of the leak-free, per-bearing validation calibration protocol.

---

## Verification Plan

### Manual Verification
*   Verify that all equations are written in standard LaTeX.
*   Ensure that no active voice/first-person pronouns ("we", "our") are used in the text, per scientific writing guidelines.
*   Confirm that all description details match the exact configurations and structures found in `src/models/mamba/hybrid_mamba.py`, `src/models/mamba/fusion_head.py`, and `src/data/dataset.py`.

---
---

# Vietnamese Version

# Kế hoạch Thực hiện: Mở rộng Bản nháp Bài báo Phát hiện Bất thường Vòng bi (Có Dàn ý Chi tiết)

Kế hoạch này chi tiết hóa việc mở rộng bản nháp bài báo nghiên cứu khoa học (`report_nckh.md` và `report_nckh_vn.md`) dựa trên cấu trúc mã nguồn thực tế trong thư mục `src/`. Nội dung bao gồm việc dọn dẹp phần Tóm tắt/Mở đầu (loại bỏ RevIN và multi-scale patching) và xây dựng cấu trúc nội dung cho Phần II (Literature Review) và Phần III (Proposed Methodology).

## Yêu cầu Người dùng Đánh giá

> [!IMPORTANT]
> **Đồng bộ hóa Kiến trúc**: Tất cả các tham chiếu đến RevIN, Z-score normalization, PatchLSTM và multi-scale patching đã bị loại bỏ khỏi codebase để chạy trực tiếp trên tín hiệu thô và đặc trưng vật lý thống kê. Dự thảo hiện tại của bài báo (`report_nckh.md`) vẫn liệt kê RevIN và multi-scale patching làm đóng góp cốt lõi.
>
> Chúng tôi sẽ cập nhật Abstract và Introduction để khớp với mã nguồn thực tế:
> *   **Loại bỏ**: Reversible Instance Normalization (RevIN).
> *   **Loại bỏ**: Multi-scale patching/decomposition (phân tách/vá mảnh đa quy mô).
> *   **Bổ sung**: Tín hiệu thô + 8 đặc trưng vật lý miền thời gian (Stats Head) nhằm định hướng mô hình với tính diễn giải vật lý cơ học.
> *   **Bổ sung**: Kiến trúc lai Mamba-CNN với khối Phân tách Chuỗi (Trend/Seasonal) và Đầu Dự báo Dung hợp (Fusion Forecasting Head).

## Các Thay đổi Đề xuất

Chúng tôi sẽ sửa đổi cả hai bản nháp tiếng Anh (`report_nckh.md`) và tiếng Việt (`report_nckh_vn.md`).

### Phần I: Dọn dẹp Abstract & Introduction
*   Cập nhật Tiêu đề bài báo (nếu cần) để bỏ cụm từ "Multi-Scale" và tập trung vào kiến trúc lai Mamba-CNN với đặc trưng vật lý.
*   Viết lại Tóm tắt (Abstract) để phản ánh việc loại bỏ RevIN/multi-scale patching, đồng thời làm nổi bật vai trò của Stats Head (8 tham số vật lý) và mô hình dự báo Hybrid Mamba-CNN.
*   Viết lại phần Mở đầu (đặc biệt là câu hỏi nghiên cứu và đóng góp cốt lõi) để đồng bộ hóa với mô hình dự báo chuỗi thời gian thô + đặc trưng vật lý.

### Phần II: Tổng quan Tài liệu (Literature Review - Phần mới)
*   **Deep Learning cho Mô hình hóa Chuỗi trong PHM**: Hạn chế của các mô hình chuỗi thời gian học sâu truyền thống đối với tín hiệu vòng bi chu kỳ dài:
    *   *LSTM*: Nút thắt cổ chai tính toán tuần tự và tiêu biến đạo hàm.
    *   *TCN*: Trường đón nhận (receptive field) bị giới hạn khi xử lý các tín hiệu có tần số lấy mẫu cực cao.
    *   *Transformers*: Độ phức tạp tính toán bậc hai ($O(N^2)$) và thiếu tính bất biến tịnh tiến cục bộ khiến mô hình dễ nhạy cảm với nhiễu nền công nghiệp.
*   **Mô hình Không gian Trạng thái Chọn lọc (Mamba) cho Chuỗi Thời gian**: Tại sao mô hình không gian trạng thái chọn lọc là lựa chọn tối ưu để theo dõi các xu hướng suy thoái cơ học dài hạn nhờ độ phức tạp tuyến tính và cơ chế quét chọn lọc (selective scan).
*   **Các Phương pháp Thiết lập Ngưỡng Động**: Nhược điểm của các ngưỡng tĩnh cố định dưới điều kiện vận hành thay đổi. Tổng quan các kỹ thuật ngưỡng động (3-Sigma, Robust/IQR, Gaussian Mixture Models và Peak Over Threshold dựa trên Lý thuyết giá trị cực biên).
*   **Literature Matrix Table**: Bảng so sánh các nghiên cứu SOTA liên quan trích xuất từ `references.md` (như FEMamba, TFG-Mamba, ModernTCN, Autoformer). (phần này sẽ bổ sung sau)
*   **Mô hình so sánh (Baselines)**: Giới thiệu chính thức các mô hình đối chứng phục vụ thực nghiệm so sánh:
    *   *LSTM*: Mạng hồi quy tiêu chuẩn cho việc mô hình hóa phụ thuộc thời gian.
    *   *TCN*: Mạng tích chập thời gian sử dụng tích chập giãn (dilated convolution).
    *   *ModernTCN*: Mô hình mạng tích chập thời gian hiện đại hóa.
    *   *PatchTST*: Mô hình Transformer dựa trên vá mảnh thời gian và độc lập kênh.
    *   *SimpleMamba*: Mô hình không gian trạng thái chọn lọc tiêu chuẩn (không tích hợp lai CNN hay Stats Head).

### Phần III: Phương pháp Đề xuất (Proposed Methodology - Phần mới)
*   **Problem Formulation**: Định nghĩa toán học chi tiết về chuỗi tín hiệu đầu vào $X$, cửa sổ lịch sử $L$, khoảng thời gian dự báo $H$, và chuỗi dự báo tái tạo $\hat{Y}$.
*   **Data Preprocessing**: Lọc thông cao Butterworth ($highpass\_freq$) để loại bỏ nhiễu động cơ nền và làm nổi bật xung va đập cơ học.
*   **Physics-Informed Stats Head**: Các công thức toán học của 8 đặc trưng thống kê vật lý được trích xuất trong `dataset.py` (Mean, Std, RMS, Peak-to-Peak, Skewness, Kurtosis, Crest Factor, Shape Factor) và cách chuẩn hóa, kết hợp chúng trong đầu dung hợp `FusionForecastHead`.
*   **Kiến trúc Hybrid Mamba-CNN**:
    *   *Phân tách chuỗi (Series Decomposition)*: Lọc trung bình trượt $\text{AvgPool1d}$ để chia thành phần xu hướng ($X_T$) và mùa vụ/xung đột biến ($X_S$).
    *   *Nhánh Xu hướng (Trend Branch)*: Giảm mẫu (downsampling) và dự báo tuyến tính.
    *   *Nhánh Mùa vụ (Seasonal Branch)*: Vá mảnh tuyến tính ($SimplePatchEmbedding$), mã hóa Mamba độc lập kênh ($MambaEncoder$), và đầu dự báo dung hợp đặc trưng vật lý ($FusionForecastHead$).
    *   *Hòa trộn tự học (Learnable Mixing)*: Hệ số hòa trộn $\alpha$ học được qua hàm Sigmoid.
*   **Tính Điểm Bất thường & Ngưỡng Quyết định**:
    *   Tính sai số dự báo MSE trong `anomaly_scorer.py`.
    *   Cơ chế ngưỡng động Peak Over Threshold (POT) dựa trên Lý thuyết giá trị cực biên (EVT) trong `metrics.py` (sử dụng Phân phối Pareto tổng quát).
    *   Các phương pháp hiệu chuẩn ngưỡng thay thế (3-Sigma, Robust/IQR, GMM).

---

## Dàn ý Chi tiết các Phần (Vietnamese)

### Phần II: Tổng quan Tài liệu
*   **A. Các mô hình chuỗi thời gian trong chẩn đoán và quản lý sức khỏe máy móc**
    *   *Mạng hồi quy (LSTM)*: Phân tích cơ chế cập nhật trạng thái lặp; thảo luận về nút thắt cổ chai khi huấn luyện song song và hiện tượng tiêu biến đạo hàm trên các chuỗi rung tần số cao.
    *   *Mạng tích chập thời gian (TCN)*: Đánh giá khả năng xử lý song song; chỉ ra nhược điểm của trường đón nhận (receptive field) cố định khi theo dõi chu trình suy thoái dài hạn.
    *   *Mạng cơ chế tự chú ý (Transformer)*: Phê phán độ phức tạp tính toán và bộ nhớ bậc hai ($O(N^2)$); giải thích sự nhạy cảm với nhiễu công nghiệp do thiếu tính bất biến dịch chuyển cục bộ.
*   **B. Mô hình không gian trạng thái chọn lọc cho dự báo chuỗi dài**
    *   Giới thiệu hệ phương trình không gian trạng thái liên tục (biểu diễn các ma trận $\mathbf{A}, \mathbf{B}, \mathbf{C}$).
    *   Cơ chế quét chọn lọc (Mamba) giúp đạt độ phức tạp tuyến tính $O(N)$ mà vẫn duy trì khả năng trích xuất phụ thuộc xa.
    *   Đánh giá các ứng dụng Mamba hiện tại trong dự đoán RUL vòng bi và chỉ ra khoảng trống nghiên cứu: thiếu các kiến trúc không giám sát dựa trên sai số dự báo (forecasting error) để tính ngưỡng phát hiện bất thường.
*   **C. Các kỹ thuật hiệu chuẩn ngưỡng phát hiện bất thường**
    *   Nhược điểm của các ngưỡng tĩnh cố định (như bội số RMS) dưới tác động của tải trọng/tốc độ motor thay đổi.
    *   Xem xét các kỹ thuật ngưỡng động/tự thích ứng: thống kê (3-Sigma, IQR), phân cụm (Gaussian Mixture Model) và mô hình hóa giá trị cực biên (Peak Over Threshold).
*   **D. Bảng ma trận định vị nghiên cứu học thuật**
    *   Bảng ma trận so sánh các mô hình SOTA về: định dạng đầu vào (thô/qua xử lý), kiến trúc mô hình, phương pháp xác định ngưỡng, độ phức tạp tính toán và hạn chế chính.
*   **E. Các mô hình đối chứng (Baseline Models) trong thực nghiệm**
    *   Cấu trúc và tham số của LSTM, TCN, ModernTCN, PatchTST (Vanilla Transformer), và SimpleMamba.
    *   Cơ sở khoa học chọn lựa các baselines này đại diện cho các trường phái: hồi quy, tích chập, chú ý (attention), và không gian trạng thái (SSM).

### Phần III: Phương pháp nghiên cứu đề xuất
*   **A. Mô tả tổng quan khung phương pháp và Phát biểu bài toán**
    *   Định nghĩa toán học của chuỗi tín hiệu đầu vào $X \in \mathbb{R}^{C \times L}$ và chuỗi dự báo đầu ra $\hat{Y} \in \mathbb{R}^{C \times H}$.
    *   Phát biểu bài toán phát hiện bất thường không giám sát dựa trên sai lệch dự báo của cửa sổ kế tiếp.
*   **B. Tiền xử lý tín hiệu và Lọc thông cao**
    *   Công thức toán học của bộ lọc thông cao Butterworth: Hàm truyền đạt $H(s)$ và dạng rời rạc hóa.
    *   Giải thích vật lý cho việc chọn tần số cắt ($highpass\_freq$): chặn nhiễu tần số thấp từ động cơ nền và duy trì các xung va đập cơ học tần số cao do nứt rỗ bề mặt vòng bi.
*   **C. Stats Head - Khối trích xuất đặc trưng vật lý thống kê**
    *   Công thức toán học và ý nghĩa vật lý cơ học của 8 đặc trưng:
        1. *Mean*: Đường cơ sở trung tâm của tín hiệu.
        2. *Standard Deviation*: Biến thiên năng lượng.
        3. *Root Mean Square (RMS)*: Chỉ số năng lượng rung động toàn phần/mức độ nghiêm trọng của lỗi.
        4. *Peak-to-Peak*: Biên độ dịch chuyển tuyệt đối lớn nhất.
        5. *Skewness*: Độ lệch bất đối xứng của phân phối rung động (chỉ thị hướng va đập).
        6. *Kurtosis*: Độ nhọn phân phối (rất nhạy cảm với các xung va đập sớm của vết nứt).
        7. *Crest Factor*: Tỷ lệ đỉnh trên RMS (nhạy cảm với tỷ lệ xung trên năng lượng nền).
        8. *Shape Factor*: Chỉ số hình dạng biên dạng sóng.
    *   Khối Chuẩn hóa lô (Batch Normalization) và lớp chiếu tuyến tính ánh xạ 8 đặc trưng vào không gian ẩn dự báo.
*   **D. Phân tách chuỗi thời gian đa quy mô**
    *   Công thức phân tách chuỗi: $X = X_{seasonal} + X_{trend}$.
    *   Trích xuất xu hướng bằng bộ lọc trung bình trượt 1 chiều: $X_{trend} = \text{AvgPool1d}(X, \text{kernel\_size})$.
    *   Chiếu tuyến tính thành phần xu hướng ra chiều dài dự báo: $\hat{Y}_{trend} = \mathbf{W}_T X_{trend} + \mathbf{b}_T$.
*   **E. Khối mã hóa Mamba độc lập kênh cho thành phần mùa vụ**
    *   *Cơ chế vá mảnh (Patching)*: Ánh xạ tuyến tính các phân đoạn cửa sổ không chồng chập vào chiều ẩn $d_{model}$ thông qua `SimplePatchEmbedding`.
    *   *Độc lập kênh (Channel Independence)*: Làm phẳng chiều batch và channel để xử lý song song độc lập, tránh rò rỉ nhiễu xuyên kênh.
    *   *Bộ mã hóa Mamba*: Cập nhật biến trạng thái và quét chọn lọc trên chuỗi patches.
    *   *Fusion Head*: Ghép nối vector biểu diễn chuỗi nén của Mamba với vector 8 đặc trưng vật lý đã chuẩn hóa của Stats Head, sau đó chiếu ra chiều dài dự báo mùa vụ.
*   **F. Khối hòa trộn tự học (Learnable Mixing)**
    *   Công thức tính toán kết quả hòa trộn cuối cùng:
        $$\alpha = \sigma(\mathbf{w}_{mix})$$
        $$\hat{Y} = \alpha \odot \hat{Y}_{seasonal} + (1 - \alpha) \odot \hat{Y}_{trend}$$
    *   Giải thích vai trò của tham số hòa trộn $\alpha$ trong việc cân bằng động giữa xu hướng suy thoái dài hạn và các xung va đập mùa vụ ngắn hạn.
*   **G. Điểm bất thường dựa trên sai số dự báo (Forecasting Error) và Cơ chế ngưỡng động POT**
    *   Công thức tính điểm bất thường: Sai số bình phương trung bình (MSE) giữa chuỗi thực tế $Y$ và chuỗi dự báo $\hat{Y}$.
    *   Công thức tính ngưỡng động Peak Over Threshold (POT) dựa trên Lý thuyết giá trị cực biên (EVT):
        *   Thiết lập ngưỡng khởi tạo phân vị cao $t$.
        *   Khớp (fit) các giá trị vượt ngưỡng $X - t$ vào Phân phối Pareto tổng quát (GPD).
        *   Tính toán ngưỡng động $z_q$ ứng với xác suất cảnh báo $q$:
            $$z_q = t + \frac{\sigma}{\gamma} \left( \left( \frac{q n}{N_t} \right)^{-\gamma} - 1 \right)$$
    *   Mô tả quy trình hiệu chuẩn ngưỡng động độc lập từng vòng bi trên tập dữ liệu lành mạnh để tránh rò rỉ thông tin kiểm thử.

---

## Kế hoạch Xác minh

### Manual Verification
*   Ensure that no active voice/first-person pronouns ("we", "our") are used in the text, per scientific writing guidelines.
*   Confirm that all description details match the exact configurations and structures found in `src/models/mamba/hybrid_mamba.py`, `src/models/mamba/fusion_head.py`, and `src/data/dataset.py`.
