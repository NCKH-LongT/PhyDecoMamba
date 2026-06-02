# HƯỚNG DẪN RÚT GỌN BÀI BÁO XUỐNG 10–12 TRANG

**Tên bài hiện tại:** *Hybrid Mamba-CNN Architecture with Physics-Informed Stats Head for Leakage-Free Bearing Anomaly Detection*  
**Mục tiêu:** Rút bản thảo hiện tại từ khoảng 24 trang xuống còn **10–12 trang** theo phong cách bài báo hội thảo/journal ngắn, giữ được đóng góp chính nhưng giảm độ dài, giảm lặp ý, giảm văn phong phóng đại.

---

## 1. Mục tiêu rút gọn

AI cần rút bài theo hướng:

- Giữ lại **ý tưởng cốt lõi** của bài báo.
- Giữ lại **đóng góp khoa học chính**.
- Giữ lại các kết quả thực nghiệm quan trọng nhất.
- Cắt bỏ phần giải thích quá dài, lặp ý, hoặc giống báo cáo kỹ thuật.
- Chuyển văn phong từ “giải thích luận văn” sang “paper học thuật ngắn gọn”.
- Không làm mất logic: **Problem → Gap → Method → Experiment → Result → Discussion → Conclusion**.

Bản rút gọn nên có độ dài mục tiêu:

| Loại bản | Số trang mục tiêu | Ghi chú |
|---|---:|---|
| Conference paper | 10 trang | Cắt mạnh Related Work và Method |
| Extended conference paper | 12 trang | Giữ thêm hình/bảng chính |
| Technical report | 20+ trang | Bản hiện tại có thể giữ làm phụ lục |

---

## 2. Thông điệp chính cần giữ

Khi rút gọn, bài chỉ nên xoay quanh 3 thông điệp chính:

### 2.1. Hybrid Mamba-CNN cho chuỗi rung động dài

Mô hình dùng kiến trúc lai **Mamba-CNN** để xử lý tín hiệu rung động vòng bi dạng chuỗi dài. Mamba giúp mô hình hóa phụ thuộc dài hạn với chi phí tuyến tính, còn CNN hỗ trợ trích xuất đặc trưng cục bộ và giảm nhiễu.

### 2.2. Physics-Informed Stats Head tăng khả năng diễn giải

Bài bổ sung một **Stats Head 8 chiều** gồm các đặc trưng thống kê/vật lý như Mean, Standard Deviation, RMS, Peak-to-Peak, Skewness, Kurtosis, Crest Factor, Shape Factor. Mục tiêu là nối biểu diễn học sâu với các chỉ số cơ học quen thuộc trong chẩn đoán vòng bi.

### 2.3. Leakage-Free POT thresholding

Bài dùng **POT-EVT** để xác định ngưỡng bất thường, nhưng chỉ hiệu chuẩn trên đoạn dữ liệu khỏe mạnh ban đầu. Đây là điểm quan trọng vì tránh việc dùng dữ liệu lỗi tương lai để chọn ngưỡng, làm tăng tính thực tế khi triển khai online.

---

## 3. Những phần cần rút mạnh

### 3.1. Abstract

**Hiện trạng:** Abstract quá dài, nhiều câu phức, chứa nhiều chi tiết kỹ thuật và baseline.  
**Yêu cầu rút:** 180–220 từ.

Abstract mới nên gồm 5 ý:

1. Bối cảnh: phát hiện bất thường vòng bi từ tín hiệu rung động.
2. Vấn đề: chuỗi dài, nhiễu, thiếu nhãn lỗi, data leakage khi đặt ngưỡng.
3. Phương pháp: Hybrid Mamba-CNN + series decomposition + Stats Head.
4. Ngưỡng: POT-EVT chỉ dùng dữ liệu healthy ban đầu.
5. Kết quả chính: cải thiện hiệu quả phát hiện và giảm chi phí tính toán.

**Cắt bỏ:**

- Các câu giải thích dài về từng baseline.
- Các cụm từ quá mạnh như “đảm bảo tuyệt đối”, “vượt trội toàn diện”, “triệt tiêu hoàn toàn”.

---

### 3.2. Introduction

**Mục tiêu độ dài:** 1.5–2 trang.

Cấu trúc nên viết lại:

1. Bearing anomaly detection quan trọng trong PHM.
2. Tín hiệu rung động có nhiễu, không dừng, khó phát hiện lỗi sớm.
3. Hạn chế của LSTM/TCN/Transformer: chuỗi dài, chi phí cao, nhạy nhiễu.
4. Mamba có tiềm năng nhưng còn thiếu tính diễn giải và thiếu thresholding chống leakage.
5. Đề xuất Hybrid Mamba-CNN + Stats Head + leakage-free POT.
6. Liệt kê contribution trong 3 bullet.

**Contribution nên giữ đúng 3 ý:**

- Đề xuất kiến trúc Hybrid Mamba-CNN có series decomposition cho bài toán phát hiện bất thường vòng bi không giám sát.
- Tích hợp Physics-Informed Stats Head 8 chiều để tăng tính diễn giải cơ học.
- Thiết kế quy trình POT-EVT không rò rỉ dữ liệu bằng cách hiệu chuẩn ngưỡng chỉ trên dữ liệu healthy ban đầu.

**Cắt bỏ:**

- Các đoạn mô tả quá rộng về PHM.
- Các câu lặp lại về “nhiễu công nghiệp nặng”.
- Các nhận định quá tuyệt đối.

---

### 3.3. Related Work

**Mục tiêu độ dài:** 1.5–2 trang.

Không nên chia quá nhiều mục nhỏ. Chỉ nên giữ 3 nhóm:

#### Nhóm 1: Deep time-series models for PHM

Gộp LSTM, TCN, Autoencoder, forecasting-based anomaly detection vào một đoạn.

#### Nhóm 2: Transformer and Mamba for long-sequence modeling

Gộp Anomaly Transformer, TimesNet, PatchTST, Mamba, FEMamba, TFG-Mamba vào một đoạn.

#### Nhóm 3: Threshold calibration and data leakage

Gộp POT, EVT, OmniAnomaly, USAD, TranAD và vấn đề chọn ngưỡng bằng test/validation chứa lỗi.

**Literature Matrix Table:**

Bảng hiện tại quá dài. Cần chuyển thành bảng rút gọn 6–8 dòng:

| Study | Method family | Key strength | Key limitation | Relation to this work |
|---|---|---|---|---|
| LSTM/TCN | Sequential/CNN | Simple sequence modeling | Limited long-range modeling | Baselines |
| PatchTST | Transformer | Efficient patching | Weak local inductive bias | Baseline |
| Mamba | SSM | Linear long-sequence modeling | Limited physical interpretability | Core encoder |
| FEMamba/TFG-Mamba | Mamba-PHM | Strong degradation modeling | Often supervised/RUL-oriented | Motivation |
| OmniAnomaly/USAD/TranAD | Reconstruction AD | Common AD baselines | Threshold leakage risk | Contrast |
| EVT/POT | Statistical thresholding | Tail modeling | Can leak if calibrated globally | Used leakage-free |

**Cắt bỏ:**

- Mô tả chi tiết từng paper 5–7 dòng.
- Các câu đánh giá quá nặng như “hoàn toàn thất bại”, “tuyệt đối không thể triển khai”.
- Những reference không trực tiếp phục vụ gap.

---

### 3.4. Methodology

**Mục tiêu độ dài:** 3–4 trang.

Method nên có 1 hình pipeline và 5 tiểu mục chính:

1. Problem Formulation
2. Series Decomposition
3. Hybrid Mamba-CNN Forecasting
4. Physics-Informed Stats Head
5. Leakage-Free POT Thresholding

#### 3.4.1. Problem Formulation

Giữ công thức đầu vào, đầu ra, anomaly score. Không giải thích quá dài.

Nên viết ngắn:

> Given a multivariate vibration sequence, the model observes a lookback window and forecasts the next horizon. The anomaly score is computed as the forecasting error between the predicted and observed signals.

#### 3.4.2. Series Decomposition

Giữ ý trend/seasonal:

- Trend: thành phần chậm, mịn, năng lượng thấp.
- Seasonal: thành phần dao động/xung, chứa nhiều thông tin lỗi.

Cắt bớt giải thích về từng hiện tượng vật lý nếu bị lặp.

#### 3.4.3. Hybrid Mamba-CNN Forecasting

Chỉ cần mô tả:

- CNN patch embedding nén chuỗi và học đặc trưng cục bộ.
- Channel independence giảm nhiễu xuyên kênh.
- Mamba block học phụ thuộc dài hạn với chi phí tuyến tính.
- Forecasting head tạo dự báo tương lai.

Không cần giải thích quá chi tiết từng tensor shape nếu không cần thiết.

#### 3.4.4. Physics-Informed Stats Head

Giữ danh sách 8 đặc trưng, nhưng không viết mỗi đặc trưng thành một đoạn dài.

Có thể chuyển thành bảng:

| Feature | Meaning |
|---|---|
| Mean | Signal offset |
| Std | Signal variability |
| RMS | Vibration energy |
| Peak-to-Peak | Impact range |
| Skewness | Distribution asymmetry |
| Kurtosis | Impulsive behavior |
| Crest Factor | Peak sharpness |
| Shape Factor | Waveform shape |

#### 3.4.5. Leakage-Free POT Thresholding

Đây là phần quan trọng, nên giữ nhưng viết gọn:

- Fit POT only on early healthy error scores.
- Use high quantile as initial threshold.
- Fit GPD to exceedances.
- Use final threshold for online anomaly decision.

Nên nhấn mạnh: không dùng future fault/test distribution để chọn threshold.

---

### 3.5. Experiments

**Mục tiêu độ dài:** 1.5–2 trang.

Nên gộp thông tin vào bảng thay vì viết dài.

#### Bảng Experimental Setup đề xuất

| Item | Setting |
|---|---|
| Dataset | Paderborn bearing dataset |
| Sensors | Two vibration channels |
| Sampling rate | 64/128 kHz depending on lifecycle stage |
| Preprocessing | Causal Butterworth high-pass filter |
| Lookback window | 4096 |
| Forecast horizon | 1024 |
| Stride | 1024 |
| Training region | Early healthy segment |
| Test region | Full lifecycle / remaining sequence |
| Baselines | LSTM, TCN, ModernTCN, PatchTST, SimpleMamba |
| Metrics | Precision, Recall, F1, AUROC, AUPRC, FAR, latency, VRAM |

#### Baseline section

Viết ngắn:

> To ensure fair comparison, all baselines are automatically scaled to a comparable parameter budget.

Không cần mô tả dài từng baseline nếu đã có bảng.

---

### 3.6. Results and Discussion

**Mục tiêu độ dài:** 2–3 trang.

Chỉ giữ các kết quả chính:

1. Main performance comparison.
2. Threshold/early detection result.
3. Efficiency result: latency, VRAM.
4. Ablation study nếu có.
5. Discussion ngắn về trade-off.

#### Cần nhấn mạnh cẩn thận

Nếu F1 của Proposed chỉ nhỉnh hơn baseline một chút, không nên viết là “vượt trội mạnh”. Nên viết:

> The proposed model achieves the best overall performance, with moderate gains in detection metrics and clearer advantages in computational efficiency and leakage-free deployment.

#### Nên thêm Ablation Study

Nếu có thể chạy thêm, nên bổ sung bảng:

| Variant | F1 | AUROC | AUPRC | FAR | Interpretation |
|---|---:|---:|---:|---:|---|
| Full model |  |  |  |  | Complete architecture |
| w/o Stats Head |  |  |  |  | Tests physical features |
| w/o CNN |  |  |  |  | Tests local filtering |
| w/o decomposition |  |  |  |  | Tests trend/seasonal split |
| Global threshold |  |  |  |  | Tests leakage-free POT |

Nếu không có ablation, nên ghi là limitation hoặc future work.

---

## 4. Hình và bảng nên giữ

### Nên giữ tối đa 4 hình

| Hình | Giữ không? | Lý do |
|---|---|---|
| Overall framework/pipeline | Bắt buộc giữ | Giải thích kiến trúc nhanh nhất |
| Series decomposition/PSD | Nên giữ, có thể gộp | Minh chứng trend/seasonal |
| Anomaly score/threshold curve | Nên giữ | Minh chứng leakage-free detection |
| Latency/VRAM comparison | Nên giữ | Thể hiện lợi thế triển khai |
| PCA/t-SNE visualization | Có thể bỏ | Phụ trợ, không bắt buộc |
| Error audit table | Nên chuyển phụ lục | Giống báo cáo kỹ thuật |

### Nên giữ 3–4 bảng

| Bảng | Giữ không? | Ghi chú |
|---|---|---|
| Compact literature matrix | Có | Rút gọn mạnh |
| Experimental setup | Có | Giúp tiết kiệm trang |
| Main performance comparison | Bắt buộc | Bảng chính |
| Efficiency comparison | Có | Quan trọng nếu accuracy gain nhỏ |
| Ablation study | Rất nên có | Làm contribution mạnh hơn |

---

## 5. Văn phong cần chỉnh

### 5.1. Tránh overclaim

Thay các cụm sau:

| Cụm hiện tại | Nên đổi thành |
|---|---|
| triệt tiêu hoàn toàn | reduce / mitigate |
| đảm bảo tuyệt đối | improve / support / enforce under the defined protocol |
| vượt trội toàn diện | achieves better overall performance |
| đường cơ sở khoa học mới | provides a practical baseline |
| thất bại hoàn toàn | remains limited / may be less effective |
| chính xác tuyệt đối | more reliable / more consistent |

### 5.2. Giảm câu quá dài

Mỗi câu nên ưu tiên 20–30 từ. Không viết một câu kéo dài 5–7 dòng.

### 5.3. Ưu tiên giọng học thuật trung tính

Không nên viết như quảng bá sản phẩm. Reviewer thường thích các câu có điều kiện:

- “The results suggest that...”
- “This indicates that...”
- “Under the evaluated setup...”
- “Compared with the selected baselines...”
- “The proposed method shows...”

---

## 6. Dàn trang mục tiêu 12 trang

| Section | Target pages |
|---|---:|
| Title + Abstract + Introduction | 2 |
| Related Work | 1.5 |
| Proposed Method | 3 |
| Experimental Setup | 1 |
| Results | 2.5 |
| Discussion + Limitations | 1 |
| Conclusion | 0.5 |
| References | 1 |
| **Total** | **12** |

---

## 7. Prompt dùng cho AI để rút bài

Dùng prompt sau để yêu cầu AI rút gọn bản thảo:

```text
You are an academic paper editor. Please shorten the uploaded manuscript into a compact 10–12 page conference-style paper.

Main goal:
- Keep the core contribution: Hybrid Mamba-CNN, Physics-Informed Stats Head, and leakage-free POT thresholding.
- Remove redundant explanations, overclaiming language, and report-like details.
- Preserve the logical structure: Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion.
- Keep important equations only. Remove excessive derivations and long physical explanations.
- Convert long experimental descriptions into compact tables.
- Reduce the literature matrix to a short comparison table.
- Keep no more than 4 key figures and 4 key tables.
- Use neutral academic English.
- Avoid words such as “absolute”, “completely eliminates”, “fully guarantees”, “state-of-the-art breakthrough” unless directly proven.

Target structure:
1. Abstract: 180–220 words.
2. Introduction: 1.5–2 pages, including 3 clear contributions.
3. Related Work: 1.5–2 pages with 3 groups only: deep time-series PHM, Transformer/Mamba models, and threshold calibration/data leakage.
4. Methodology: 3–4 pages covering problem formulation, series decomposition, Hybrid Mamba-CNN, Stats Head, and leakage-free POT.
5. Experiments: 1–1.5 pages using a compact setup table.
6. Results and Discussion: 2–3 pages focusing on main performance, efficiency, thresholding, and ablation if available.
7. Conclusion: 0.5 page.

Important editing rules:
- Do not remove the novelty claims, but rewrite them more carefully.
- If the performance gain over baselines is small, do not overstate accuracy improvement. Instead, emphasize the combined value of accuracy, efficiency, interpretability, and leakage-free deployment.
- If ablation results are missing, add a short limitation sentence suggesting future ablation studies.
- Keep references only if they are cited in the shortened text.
- Ensure all citations are formatted consistently.

Output:
- Return a revised paper draft in academic English.
- After the draft, provide a short list of what was removed and why.
```

---

## 8. Checklist sau khi AI rút xong

Sau khi có bản rút gọn, kiểm tra các điểm sau:

- [ ] Abstract dưới 220 từ.
- [ ] Introduction có đúng 3 contribution rõ ràng.
- [ ] Related Work không vượt 2 trang.
- [ ] Literature matrix đã rút còn tối đa 8 dòng.
- [ ] Method không giải thích lan man từng công thức.
- [ ] Experimental setup có bảng tóm tắt.
- [ ] Results có bảng so sánh chính.
- [ ] Có discussion về accuracy gain nếu không quá lớn.
- [ ] Có limitation nếu thiếu ablation hoặc tập dữ liệu còn nhỏ.
- [ ] Không còn các cụm overclaim.
- [ ] Reference không còn mục chưa hoàn chỉnh như “Author et al.”.
- [ ] Tổng độ dài khoảng 10–12 trang.

---

## 9. Gợi ý tiêu đề rút gọn

Có thể đổi tiêu đề ngắn hơn:

**Leakage-Free Bearing Anomaly Detection with Hybrid Mamba-CNN and Physics-Informed Statistical Features**

Hoặc:

**A Physics-Informed Hybrid Mamba-CNN for Leakage-Free Bearing Anomaly Detection**

Tiêu đề nên ngắn hơn bản hiện tại nhưng vẫn giữ 3 keyword quan trọng: **Hybrid Mamba-CNN**, **Physics-Informed**, **Leakage-Free Bearing Anomaly Detection**.
