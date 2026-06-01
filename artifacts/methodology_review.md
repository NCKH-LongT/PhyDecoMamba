# Phản biện & Đề xuất Viết lại phần Methodology (Chuẩn Q1 Journal)

Tài liệu này chứa các phân tích phản biện chi tiết, chỉ ra các điểm hợp lý, bất hợp lý và đề xuất viết lại (bằng cả Tiếng Việt và Tiếng Anh) cho phần **Methodology (Phương pháp nghiên cứu)** của bài báo *"Hybrid Mamba-CNN Architecture with Physics-Informed Stats Head for Leakage-Free Bearing Anomaly Detection"*.

---

## I. Tóm tắt phản biện & Điểm cần chỉnh sửa lớn (Reviewer Objections)

### 1. Điểm hợp lý (Strengths)
* **Kịch bản thực tế:** Học không giám sát bằng cách huấn luyện mô hình dự báo cửa sổ kế tiếp trên dữ liệu khỏe mạnh (`condition == 0`) rất thực tế.
* **Tính diễn giải kỹ thuật:** Tích hợp Stats Head (RMS, Kurtosis, Crest Factor, Shape Factor,...) bổ khuyết cho tính chất "hộp đen" của Mamba, tăng độ tin cậy cơ học.
* **Chống rò rỉ dữ liệu:** POT (Peak Over Threshold) được hiệu chuẩn cục bộ chỉ trên dữ liệu khỏe mạnh giúp mô hình sẵn sàng triển khai trực tuyến (online) thực tế.

### 2. Điểm bất hợp lý & Rủi ro học thuật (Academic Risks)
* **Mâu thuẫn giữa Lọc thông cao (High-pass) và Phân tách chuỗi (Decomposition):** 
  * *Rủi ro:* Lọc thông cao Butterworth triệt tiêu toàn bộ tần số thấp. Sau đó, khối Phân tách chuỗi lại dùng trung bình trượt (Moving Average - lọc thông thấp) trên tín hiệu đã lọc để tìm xu hướng ($X_{trend}$). Điều này khiến $X_{trend} \approx 0$ (chuỗi toàn số 0 hoặc nhiễu trắng), nhánh tuyến tính dự báo xu hướng trở nên vô dụng.
* **Sự thiếu nhất quán về Dung hợp kênh (Cross-Channel Fusion):**
  * *Rủi ro:* Bảng so sánh (Table I) ghi mô hình đề xuất *"fuses channels within the Fusion Head"*. Nhưng trong phương trình toán học ở phần Methodology, cả nhánh Mamba, Stats Head và Mixing Module đều chạy độc lập kênh ($c = 1 \dots C$). Không có phương trình nào thể hiện việc trộn thông tin xuyên kênh.
* **Vấn đề trễ pha (Phase Delay) của bộ lọc tiền xử lý:**
  * *Rủi ro:* Nếu bộ lọc Butterworth dùng trong ứng dụng thực tế (online), nó phải là bộ lọc nhân quả (causal), từ đó gây ra trễ pha phụ thuộc tần số, làm giảm độ chính xác của dự báo chuỗi thời gian. Nếu dùng bộ lọc phi nhân quả (như `filtfilt` không trễ pha), mô hình sẽ bị rò rỉ dữ liệu tương lai (vi phạm nguyên tắc "leakage-free").

---

## II. Đề xuất Viết lại - Bản Tiếng Việt (`report_nckh_vn.md`)

Dưới đây là bản thảo đề xuất viết lại phần Methodology bằng Tiếng Việt.
* Các phần đề xuất thay đổi lớn được đặt trong khối **`> 📌 [ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION]`** kèm theo phần chữ **in đậm** để dễ đối chiếu.

### III. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

#### A. Phát biểu Bài toán và Khung Phương pháp Tổng quan
Quá trình giám sát sức khỏe cấu trúc của vòng bi công nghiệp thông qua dữ liệu chuỗi thời gian được định hình dưới kịch bản phát hiện bất thường không giám sát dựa trên cơ chế dự báo cửa sổ kế tiếp. Giả định một chuỗi tín hiệu rung động đa biến thu thập từ hệ thống cảm biến gia tốc gồm $C$ kênh đo (đại diện cho các hướng thu thập dữ liệu cơ học như trục ngang và trục dọc) với chiều dài lịch sử quan sát (lookback window) là $L$. Tensor dữ liệu đầu vào tại thời điểm $t$ được định nghĩa là $X_t \in \mathbb{R}^{C \times L}$. Mục tiêu toán học của hệ thống là dự báo chính xác chuỗi tín hiệu tương lai trong một cửa sổ dự báo (forecast horizon) có độ dài $H$, ký hiệu là $\hat{Y}_t \in \mathbb{R}^{C \times H}$, nhằm đối chiếu với chuỗi tín hiệu thực tế tương ứng $Y_t \in \mathbb{R}^{C \times H}$.

Khung phương pháp đề xuất vận hành theo nguyên lý học biểu diễn không giám sát, trong đó toàn bộ quá trình tối ưu hóa mạng chỉ sử dụng dữ liệu thuộc pha hoạt động lành mạnh ban đầu của thiết bị—được xác định bởi các nhãn chỉ thị trạng thái bằng không ($\text{condition} = 0$). Cơ sở khoa học của phương pháp này dựa trên giả thuyết rằng mô hình sẽ thiết lập một không gian ẩn tối ưu để tái cấu trúc và dự báo các đặc tính động học bình thường của hệ thống cơ khí. Khi vòng bi xuất hiện các vết nứt rỗ, mài mòn hoặc các dạng tổn thất cấu trúc (fault state), sự xuất hiện của các xung va đập chuyển tiếp phi tuyến tính và sự dịch chuyển phân phối biên sẽ phá vỡ tính quy luật của chuỗi dữ liệu. Hệ quả là sai lệch bình phương giữa tín hiệu thực tế $Y_t$ và tín hiệu dự báo $\hat{Y}_t$ sẽ tăng vọt, cung cấp một đường cơ sở toán học đáng tin cậy để thiết lập điểm số bất thường (anomaly score) và kích hoạt ranh giới cảnh báo trực tuyến mà không phụ thuộc vào nguồn dữ liệu nhãn lỗi khan hiếm.

#### B. Tiền xử lý Tín hiệu qua Bộ lọc Thông cao Butterworth **Nhân quả**

> 📌 **[ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION - SỬA LỖI LOGIC BỘ LỌC VÀ TRỄ PHA]**
> *(Giải thích: Bổ sung tính chất nhân quả rời rạc qua bilinear transformation và giải trình trễ pha tự thích ứng để thuyết phục reviewer về tính chất không rò rỉ dữ liệu).*
> 
> Để triệt tiêu các thành phần dao động tần số thấp phát sinh từ động cơ nền, nhiễu môi trường hoặc các đặc tính cơ học không liên quan đến sự suy thoái của vòng bi, tín hiệu rung động thô ban đầu được truyền qua một bộ lọc thông cao Butterworth **nhân quả** bậc $N$. Hàm truyền đạt bình phương biên độ của bộ lọc trong miền tần số liên tục được biểu diễn dưới dạng:
> 
> $$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega_c}{\omega}\right)^{2N}}$$
> 
> Trong đó $\omega_c$ ký hiệu tần số cắt cấu hình, và $\omega$ là tần số thành phần của tín hiệu. **Để triển khai bộ lọc toán học này trực tiếp vào luồng xử lý dữ liệu thời gian thực rời rạc mà không gây rò rỉ thông tin tương lai, phép biến đổi song tuyến tính (bilinear transformation) được áp dụng nhằm chuyển đổi hàm truyền đạt từ miền tần số liên tục sang miền thời gian rời rạc, thiết lập phương trình sai phân hiệu chỉnh có dạng:**
> 
> **$$y[t] = \sum_{i=0}^N b_i x[t-i] - \sum_{i=1}^N a_i y[t-i]$$**
> 
> **Trong đó $x$ và $y$ lần lượt là tín hiệu trước và sau khi lọc tại bước thời gian $t$, trong khi $b_i$ và $a_i$ là hệ số bộ lọc được xác định qua tần số lấy mẫu $f_s$. Nhờ thiết lập nhân quả này, độ trễ pha sinh ra từ bộ lọc sẽ được mô hình học sâu học cách bù đắp một cách tự nhiên trong quá trình tối ưu hóa dự báo.** Ý nghĩa vật lý của cấu phần tiền xử lý này là cô lập và bảo toàn các vi xung động va đập biên độ nhỏ ở dải tần số cao—vốn là chỉ thị nhạy cảm nhất đối với hiện tượng bong tróc hoặc rỗ bề mặt thớ cơ học ở giai đoạn chớm lỗi—đồng thời ổn định hóa phân phối dữ liệu đầu vào.

#### C. Khối Phân tách Chuỗi Thời gian Thích ứng (Series Decomposition)

> 📌 **[ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION - HOÀN THIỆN TOÁN HỌC KHỐI PHÂN TÁCH CHUỖI]**
> *(Giải thích: Đồng bộ với nguyên lý tối giản kiến trúc từ DMamba, đưa ra các công thức toán học bị khuyết trong file gốc và giữ nguyên việc phân tách trung bình trượt trên chuỗi tín hiệu đã lọc $X$ directly).*
> 
> Đồng bộ với nguyên lý tối giản kiến trúc (architectural parsimony) được giới thiệu trong các mô hình phân tách chuỗi thời gian tiên tiến như kiến trúc DMamba [22], khung phương pháp đề xuất thực hiện tách biệt hoàn toàn quy trình xử lý các luồng thành phần mùa vụ (seasonal) và xu hướng (trend). Bản chất của cơ chế này là phân rã chuỗi tín hiệu đã lọc $X$ thành hai thành phần có đặc tính vật lý và thống kê riêng biệt nhằm tối ưu hóa hiệu suất biểu diễn của mạng học sâu. Quy trình phân tách được thực thi thông qua toán tử trung bình trượt một chiều ($\text{AvgPool1d}$) trượt dọc theo trục thời gian lịch sử:
> 
> **$$X_{trend} = \text{AvgPool1d}(\text{Pad}(X))$$**
> 
> **$$X_{seasonal} = X - X_{trend}$$**
> 
> Trong đó $2r+1$ là kích thước nhân lọc trung bình trượt. Lớp đệm ($\text{Pad}$) áp dụng cơ chế lặp biên để đảm bảo tensor xu hướng $X_{trend} \in \mathbb{R}^{C \times L}$ duy trì tính đồng nhất về kích thước hình học với chuỗi gốc.
> 
> Thành phần xu hướng $X_{trend}$ đại diện cho đường trung bình dịch chuyển, phản ánh quá trình tiến triển mài mòn cơ học tần số thấp, chuyển động chậm theo thời gian và có độ phức tạp chiều thấp. Ngược lại, thành phần mùa vụ $X_{seasonal}$ cô lập các biến động động học phi tuyến tính mạnh, bao gồm các chu kỳ quay đồng bộ của trục máy và các chuỗi xung va đập chuyển tiếp tần số cao gây ra bởi hư tổn cấu trúc vòng bi. Việc tách biệt này ngăn chặn hiện tượng các thành phần xu hướng năng lượng cao làm lu mờ các vi xung đột biến mùa vụ, cho phép các nhánh mạng chuyên biệt tập trung vào các miền đặc trưng tương thích.

#### D. Nhánh Dự báo Xu hướng Tuyến tính
Mặc dù thể hiện xu hướng suy thoái lũy tiến, thành phần $X_{trend}$ có độ phức tạp chiều rất thấp và quy luật biến thiên mịn. Việc chuyển luồng dữ liệu này qua các bộ mã hóa có tham số phức tạp như mạng attention hay quét không gian trạng thái chọn lọc là không cần thiết, dễ dẫn đến hiện tượng quá khớp (overfitting) và bùng nổ chi phí tính toán. Do đó, một nhánh chiếu tuyến tính trực tiếp (linear projection) được cấu trúc để dự báo thành phần xu hướng tương lai $\hat{Y}_{trend}$ từ cửa sổ lịch sử:

$$\hat{Y}_{trend} = W_{trend} \cdot \text{Downsample}(X_{trend}) + b_{trend}$$

Trong đó $W_{trend}$ đại diện cho ma trận trọng số chiếu, và $b_{trend}$ là vector định thiên (bias). Để tối ưu hóa hiệu suất lưu trữ và tăng cường tính ổn định cho các hệ thống có cửa sổ quan sát lịch sử siêu dài, một toán tử giảm mẫu thích ứng ($\text{Downsample}$) với bước nhảy $S$ được tích hợp để nén chuỗi xu hướng trước khi thực hiện phép nhân ma trận. Ma trận trọng số $W_{trend}$ được thiết lập dưới ràng buộc chia sẻ trọng số độc lập kênh (channel-independent weight sharing), nghĩa là một ma trận duy nhất được áp dụng đồng nhất cho mọi kênh cảm biến $c$, giúp giảm thiểu đáng kể số lượng tham số cần huấn luyện và bảo toàn xu hướng suy thoái cơ học chung.

#### E. Nhánh Dự báo Mùa vụ dựa trên Mamba-CNN Độc lập Kênh
Thành phần mùa vụ $X_{seasonal}$ chứa đựng các thông tin động học phi tuyến tính phức tạp cấu thành từ nhiễu và xung lỗi, được định tuyến qua một nhánh xử lý lai kết hợp cấu trúc CNN cục bộ và mạng không gian trạng thái chọn lọc Mamba nhằm khai thác tối đa ngữ cảnh thời gian.

##### 1) Cơ chế Vá mảnh Thời gian dạng Tích chập (CNN Patch Embedding)
Để nén chiều dài chuỗi đầu vào và tăng cường thiên kiến quy nạp không-thời gian cục bộ, chuỗi thành phần mùa vụ được phân tách thành các mảnh thời gian (patches) kích thước $P$ với bước nhảy dịch chuyển $S_{patch}$. Số lượng mảnh thời gian $M$ được xác định qua công thức toán học rời rạc:

$$M = \left\lfloor \frac{L - P}{S_{patch}} \right\rfloor + 1$$

Quá trình trích xuất và chiếu các mảnh này vào không gian ẩn chiều $D$ được thực thi song song bằng một tầng tích chập một chiều ($\text{Conv1d}$) trượt dọc theo trục thời gian của chuỗi với số kênh đầu ra bằng $D$, kích thước nhân lọc bằng $P$ và bước nhảy bằng $S_{patch}$:

$$Z_{patch} = \text{Conv1d}(X_{seasonal})$$

Ý nghĩa vật lý của khối nhúng tích chập này là vận hành như một bộ lọc thông dải cục bộ thích ứng, giúp làm mịn dữ liệu và nén toàn bộ các mẫu xung biến động động học trong một thời gian ngắn thành một vector đại diện có mật độ thông tin cao.

##### 2) Cơ chế Độc lập Kênh (Channel Independence - CI)
Nhằm triệt tiêu hiện tượng rò rỉ và lan truyền nhiễu xuyên kênh giữa các trục cảm biến khác nhau, chiều kích thước lô huấn luyện ($B$) và chiều kênh ($C$) được làm phẳng để chuyển đổi cấu trúc tensor không gian ẩn:

$$Z_{patch} \in \mathbb{R}^{B \times C \times D \times M} \rightarrow Z_{flat} \in \mathbb{R}^{(B \cdot C) \times D \times M}$$

Ràng buộc này buộc bộ mã hóa phía sau phải xử lý dữ liệu từ mỗi cảm biến như một thực thể độc lập chuỗi đơn, bảo toàn nguyên vẹn các đặc tính động học đặc trưng của từng trục đo cơ học.

##### 3) Khối lai Mamba-CNN
Vectơ ẩn $Z_{flat}$ sau đó được truyền qua các khối mã hóa không gian trạng thái chọn lọc lai tích chập. Tại mỗi khối, tín hiệu chiếu ẩn $H_{in}$ tại bước thời gian mảnh $m$ được dẫn qua một nhánh tích chập một chiều cục bộ (Local 1D CNN) với kích thước nhân lọc $K$ và hàm kích hoạt phi tuyến SiLU để triệt tiêu nhiễu đo lường ngẫu nhiên:

$$H_{conv} = \text{SiLU}(\text{Conv1d}(H_{in}))$$

Chuỗi đặc trưng đã làm sạch $H_{conv}$ đóng vai trò là toán tử điều hướng đầu vào cho mô hình không gian trạng thái tuyến tính chọn lọc ($\text{SSM}$), thực hiện cập nhật trạng thái ẩn liên tục dựa trên các ma trận hệ số biến đổi phụ thuộc dữ liệu:

$$h_t = A h_{t-1} + B H_{conv}[t]$$

$$H_{mamba} = C h_t + D H_{conv}[t]$$

Việc tích hợp lớp tích chập CNN ngay trước bước quét tuyến tính của Mamba thiết lập một bộ lọc nhiễu cục bộ vững chắc, ngăn chặn hiện tượng bão hòa trạng thái ẩn của mô hình khi đối mặt với mật độ nhiễu trắng công nghiệp lớn.

##### 4) Đầu dự báo mùa vụ (Seasonal Forecasting Head)
Biểu diễn context toàn cục sau khi quét qua mạng Mamba, ký hiệu là $Z_{global}$, được chuyển đến đầu dự báo chuyên dụng để chiếu ngược về không gian miền thời gian của cửa sổ dự báo tương lai:

$$\hat{Y}_{seasonal} = W_{forecasting} \cdot Z_{global} + b_{forecasting}$$

#### F. Khối Trích xuất Đặc trưng Vật lý Thống kê Dẫn đường (Stats Head)

> 📌 **[ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION - TRÍCH XUẤT TỪ TÍN HIỆU THÔ VÀ HOÀN THIỆN CÔNG THỨC]**
> *(Giải thích: Khẳng định các đặc trưng như trị trung bình $\mu$ phải lấy từ chuỗi rung động thô gốc $X_{raw}$ trước khi lọc thông cao Butterworth, đồng thời hoàn thiện toàn bộ các phương trình toán học bằng ký hiệu LaTeX chuẩn xác).*
> 
> Để bổ khuyết cho các không gian ẩn hướng dữ liệu thuần túy vô hướng vốn vận hành như các hộp đen, một khối Stats Head vật lý được cấu trúc nhằm nhúng các bộ mô tả cơ học tường minh vào mạng. Khối này thực hiện trích xuất một vectơ đặc trưng thống kê miền thời gian 8 chiều từ **cửa sổ lịch sử thô ban đầu $X_{raw} \in \mathbb{R}^{C \times L}$** dựa trên hệ phương trình toán học cơ học:
> 
> * **Giá trị Trung bình (Mean)** (Chỉ thị độ lệch tâm và dịch chuyển hằng số một chiều):
>   $$\mu = \frac{1}{L} \sum_{i=1}^L X_{raw}[i]$$
> * **Độ lệch Chuẩn (Standard Deviation)** (Đại diện cho biên độ biến động năng lượng xung quanh trị trung bình):
>   $$\sigma = \sqrt{\frac{1}{L} \sum_{i=1}^L (X_{raw}[i] - \mu)^2}$$
> * **Giá trị Hiệu dụng (Root Mean Square - RMS)** (Chỉ thị cốt lõi về tổng năng lượng phá hủy cấu trúc vật lý):
>   $$\text{RMS} = \sqrt{\frac{1}{L} \sum_{i=1}^L (X_{raw}[i])^2}$$
> * **Biên độ Đỉnh-Đỉnh (Peak-to-Peak)** (Phạm vi va đập và giới hạn biên độ tuyệt đối của dao động):
>   $$X_{p-p} = \max(X_{raw}) - \min(X_{raw})$$
> * **Độ lệch Phân phối (Skewness)** (Đo lường tính bất đối xứng của mật độ phân phối dữ liệu, nhạy cảm với vết nứt sớm):
>   $$\text{Skew} = \frac{1}{L \cdot \sigma^3} \sum_{i=1}^L (X_{raw}[i] - \mu)^3$$
> * **Độ nhọn Phân phối (Kurtosis)** (Chỉ thị nhạy cảm bậc nhất đối với các xung va đập đột biến khi xuất hiện lỗi rỗ bề mặt):
>   $$\text{Kurt} = \frac{1}{L \cdot \sigma^4} \sum_{i=1}^L (X_{raw}[i] - \mu)^4$$
> * **Hệ số Đỉnh (Crest Factor)** (Tỷ số giữa giá trị đỉnh tuyệt đối và giá trị hiệu dụng, biểu thị độ nhọn của xung va đập):
>   $$\text{CF} = \frac{\max(|X_{raw}|)}{\text{RMS}}$$
> * **Hệ số Hình dạng (Shape Factor)** (Tỷ số giữa giá trị RMS và giá trị trung bình tuyệt đối, phản ánh biên dạng của sóng tín hiệu):
>   $$\text{SF} = \frac{\text{RMS}}{\frac{1}{L} \sum_{i=1}^L |X_{raw}[i]|}$$
> 
> Vectơ đặc trưng vật lý 8 chiều $V_{stats} \in \mathbb{R}^{C \times 8}$ này được chuẩn hóa qua tầng BatchNormalization và chuyển đổi tuyến tính trước khi thực hiện phép nối (concatenate) trực tiếp với vectơ không gian ẩn của luồng Mamba:
> 
> $$Z_{fused} = \text{Concat}(Z_{global}, \text{Linear}(V_{stats}))$$
> 
> Cấu trúc này mang tính modulable linh hoạt, được điều khiển bởi tham số cấu hình hệ thống $\text{use\_stats}$. Nếu cấu hình $\text{use\_stats} = \text{False}$, luồng trích xuất này sẽ tự động được ngắt bỏ để trả về nhánh dự báo Mamba mùa vụ nguyên bản.

#### G. Khối Hòa trộn Hai Nhánh Tự học (Learnable Dual-Stream Mixing Module)
Để vượt qua các giới hạn của phép cộng tích hợp trực tiếp vốn cố định tỷ trọng đóng góp của các thành phần, khung phương pháp đề xuất triển khai một mô-đun hòa trộn tự học dựa trên hệ số trọng số động $\alpha_c$ được tối ưu hóa độc lập cho từng kênh cảm biến $c$:

$$\hat{Y}_c = \sigma(\alpha_c) \hat{Y}_{trend, c} + (1 - \sigma(\alpha_c)) \hat{Y}_{seasonal, c}$$

Trong đó $\sigma$ đại diện cho hàm kích hoạt Sigmoid toán học nhằm ràng buộc chặt chẽ miền giá trị của hệ số trong khoảng $(0, 1)$, và $\alpha_c$ là tham số có khả năng tự cập nhật đạo hàm trong quá trình lan truyền ngược tương ứng với mỗi kênh cảm biến.

Cơ chế hòa trộn tự học này cho phép mạng tối ưu hóa tự động điều chỉnh linh hoạt tỷ trọng đóng góp giữa xu hướng năng lượng mài mòn dài hạn (thành phần trend) và các dao động biến động xung kích đột biến ngắn hạn (thành phần seasonal) dựa trên đặc tính vật lý riêng biệt của từng vị trí đặt cảm biến gia tốc, tối ưu hóa độ trung thực của chuỗi dự báo tổng hợp $\hat{Y}$.

#### H. Tính Điểm Bất thường và Xác định Ngưỡng động POT-EVT không Rò rỉ Dữ liệu
Sau khi thu được chuỗi dự báo tổng hợp $\hat{Y}_t$, điểm số bất thường (Anomaly Score) tại mỗi cửa sổ thời gian $t$ được định nghĩa chính thức bằng sai số bình phương trung bình (Mean Squared Error - MSE) trên toàn bộ các cảm biến và bước thời gian của horizon $H$:

$$S_t = \frac{1}{C \cdot H} \sum_{c=1}^C \sum_{h=1}^H (Y_{t, c, h} - \hat{Y}_{t, c, h})^2$$

Để thiết lập ranh giới quyết định động trực tuyến một cách khách quan, Lý thuyết giá trị cực biên (Extreme Value Theory - EVT) thông qua kỹ thuật vượt ngưỡng (Peak Over Threshold - POT) được tích hợp trực tiếp vào pha suy luận của mô hình. Quy trình tính toán tuân thủ nghiêm ngặt ràng buộc không rò rỉ dữ liệu (leakage-free) bằng cách chỉ thực hiện khớp hàm mật độ xác suất phân phối trên tập sai số dự báo thuộc phân đoạn dữ liệu khỏe mạnh lịch sử ban đầu, cô lập hoàn toàn khỏi mọi thông tin suy thoái tương lai. Tiến trình thực thi gồm các bước toán học rời rạc:

1. Xác định một ngưỡng neo ban đầu $th_0$ bằng cách trích xuất phân vị cao (ví dụ: $98\%$) của chuỗi sai số dự báo thu được từ tập dữ liệu vận hành lành mạnh cơ sở.
2. Lọc và thu thập tập hợp các giá trị vượt ngưỡng cực biên dương: $E_t = \{S_t - th_0 \mid S_t > th_0\}$.
3. Khớp tập hợp các giá trị vượt ngưỡng $E_t$ vào Phân phối Pareto Tổng quát (Generalized Pareto Distribution - GPD) nhằm mô tả toán học chính xác phân phối phần đuôi của sai số:
   $$F(x) = 1 - \left(1 + \frac{\gamma x}{\beta}\right)^{-\frac{1}{\gamma}}$$
   Trong đó $\gamma$ và $\beta$ lần lượt là tham số hình dáng (shape parameter) và tham số quy mô (scale parameter) được tối ưu hóa qua phương pháp ước lượng hợp lý cực đại (Maximum Likelihood Estimation - MLE).
4. Tính toán ngưỡng quyết định động cuối cùng $z_q$ ứng với một xác suất mục tiêu cảnh báo cực nhỏ $q$ (ví dụ: $q = 10^{-4}$):
   $$z_q = th_0 + \frac{\beta}{\gamma} \left( \left( \frac{n \cdot q}{N_z} \right)^{-\gamma} - 1 \right)$$
   Trong đó $n$ đại diện cho tổng số lượng mẫu quan sát cơ sở, và $N_z$ là số lượng mẫu thực tế vượt qua ngưỡng neo $th_0$ ban đầu.

Trong giai đoạn giám sát trực tuyến thời gian thực, bất kỳ cửa sổ thời gian nào có điểm số bất thường thỏa mãn điều kiện logic $S_t > z_q$ sẽ lập tức được hệ thống phân loại là trạng thái bất thường cấu trúc, đảm bảo tính thực tiễn cao và triệt tiêu hoàn toàn tỷ lệ báo động giả do nhiễu động vận hành.

---

## III. Đề xuất Viết lại - Bản Tiếng Anh (`report_nckh.md`)

* Các phần đề xuất thay đổi lớn được đặt trong khối **`> 📌 [ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION]`** kèm theo phần chữ **in đậm** bằng Tiếng Anh.

### III. PROPOSED METHODOLOGY

#### A. Problem Formulation and Framework Overview
The structural health monitoring of industrial rolling element bearings utilizing time-series trajectories is formulated under an unsupervised anomaly detection paradigm driven by a next-window forecasting mechanism. Let $X_t \in \mathbb{R}^{C \times L}$ denote a multivariate vibration sequence collected from an acceleration sensor array across $C$ physical channels (representing distinct spatial measurement axes, such as horizontal and vertical directions) over a historical lookback window of length $L$. The mathematical objective of the framework is to forecast the subsequent operational sequence within a defined forecast horizon of length $H$, denoted as $\hat{Y}_t \in \mathbb{R}^{C \times H}$, which is sequentially evaluated against the corresponding ground-truth future target sequence $Y_t \in \mathbb{R}^{C \times H}$.

The proposed architecture operates strictly under the principles of unsupervised representation learning, wherein network optimization is executed exclusively utilizing data distributions derived from the initial early-stage healthy operational phases of the machinery—identified via binary condition indicator variables equal to zero ($\text{condition} = 0$). The theoretical foundation of this approach relies on the hypothesis that the network will establish an optimal latent space characterizing the normal cyclical dynamics of the mechanical asset. Upon the onset of structural degradation profiles, such as surface pitting, spalling, or localized wear (fault states), the emergence of non-linear transient impact shocks and severe distribution shifts disrupts the learned temporal regularities. Consequently, the squared discrepancy between the ground-truth future sequence $Y_t$ and the reconstructed forecast sequence $\hat{Y}_t$ escalates sharply, yielding a robust mathematical baseline to compute anomaly scores and trigger online warning boundaries without requiring scarce or expensive labeled fault repositories.

#### B. Signal Preprocessing via **Causal** Butterworth High-Pass Filter

> 📌 **[ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION - SIGNAL FILTERING & PHASE DELAY]**
> *(Clarification: Replaced offline zero-phase filtering with discrete causal filtering via bilinear transformation to represent a realistic online scenario and address phase delays).*
> 
> To suppress low-frequency operational oscillations originating from baseline induction motors, environmental ambient noise, or extraneous mechanical dynamics unrelated to bearing degradation, the raw vibration sequences are initially passed through an $N$-th order **causal** high-pass Butterworth filter. The magnitude squared frequency response of the filter in the continuous frequency domain is mathematically expressed as follows:
> 
> $$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega_c}{\omega}\right)^{2N}}$$
> 
> where $\omega_c$ denotes the configured cutoff frequency parameter, and $\omega$ represents the component frequency of the signal execution. **To implement this continuous mathematical operator directly into a real-time discrete data processing pipeline without future data leakage, a bilinear transformation is applied to map the transfer function from the continuous $s$-domain to the discrete $z$-domain, establishing a discrete time-domain difference equation of the form:**
> 
> **$$y[t] = \sum_{i=0}^N b_i x[t-i] - \sum_{i=1}^N a_i y[t-i]$$**
> 
> **where $x$ and $y$ denote the discrete signals immediately prior to and following the filtering operation at time step $t$, respectively, while $b_i$ and $a_i$ represent the filter coefficients derived from the sensor sampling frequency $f_s$. Under this causal formulation, the phase delay introduced by the filter is naturally learned and compensated for by the downstream deep sequence forecaster.** The physical rationale of this preprocessing stage is to isolate and preserve low-amplitude, high-frequency transient impact impulses—which serve as the most sensitive early diagnostic indicators of surface fatigue cracking—while simultaneously stabilizing the input data distribution.

#### C. Distribution-Adaptive Series Decomposition Block

> 📌 **[ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION - DECOMPOSITION BLOCK MATH COMPLETENESS]**
> *(Clarification: Aligns with the architectural parsimony principles of DMamba, completing the missing mathematical formulations in the original document while preserving the standard moving average decomposition on the filtered signal $X$).*
> 
> Consistent with the structural parsimony verified by DMamba [22], the proposed framework separates the modeling pipelines of seasonal and trend components. This mechanism decomposes the filtered signal $X$ into two components with distinct physical and statistical behaviors:
> 
> **$$X_{trend} = \text{AvgPool1d}(\text{Pad}(X))$$**
> 
> **$$X_{seasonal} = X - X_{trend}$$**
> 
> where $2r+1$ represents the moving average pooling kernel size. An edge-replication padding layer ($\text{Pad}$) is enforced to guarantee that the trend tensor $X_{trend} \in \mathbb{R}^{C \times L}$ maintains strict geometric dimensionality alignment with the original input.
> 
> The trend component $X_{trend}$ represents the moving average, capturing slow-moving, low-frequency degradation pathways over time. Concurrently, the seasonal component $X_{seasonal}$ isolates non-linear high-frequency variations and impact shock cycles. This division prevents high-energy trend profiles from masking subtle seasonal impulse changes, allowing specialized networks to focus on compatible feature domains.

#### D. Linear Trend Forecasting Stream
Consistent with the structural parsimony verified by DMamba, the trend component $X_{trend}$ models the smooth, low-frequency trend of bearing wear. Routing this low-complexity stream through highly parameterized attention layers or selective state space scanning matrices is structurally redundant, frequently inducing overfitting and unnecessary computational overhead. Consequently, a direct linear projection layer is structured to forecast the future trend profile $\hat{Y}_{trend}$ across the target horizon:

$$\hat{Y}_{trend} = W_{trend} \cdot \text{Downsample}(X_{trend}) + b_{trend}$$

where $W_{trend}$ denotes the projection weight matrix, and $b_{trend}$ represents the bias vector. To optimize memory resource footprints and enhance training numerical stability under extended lookback windows, an adaptive downsampling pooling layer ($\text{Downsample}$) with a configured stride $S$ is integrated to compress the trend sequence prior to the execution of the matrix multiplication. The projection parameter matrix $W_{trend}$ is regularized under a strict channel-independent weight sharing constraint, meaning a singular weight configuration is applied uniformly across all $C$ sensor channels, drastically decreasing the global trainable parameter volume while preserving the underlying unified degradation path.

#### E. Seasonal Mamba-CNN Forecasting Stream
The seasonal component $X_{seasonal}$ encapsulates complex non-linear temporal variations constituted by background operational noise and fault-induced impulses, which are routed through a hybrid processing stream fusing localized convolutional blocks and selective Mamba state space encoders to thoroughly map long-range dependencies.

##### 1) Convolutional Patch Embedding Mechanism (CNN Patch Embedding)
To reduce the token sequence length forwarded to the recurrent layers and reinforce localized spatial-temporal inductive biases, the seasonal time-series is partitioned into localized patches of size $P$ with a sliding stride of $S_{patch}$. The total volume of generated temporal tokens $M$ is determined via the discrete mathematical expression:

$$M = \left\lfloor \frac{L - P}{S_{patch}} \right\rfloor + 1$$

The extraction and projection of these patches into a latent embedding space of dimension $D$ are executed in parallel utilizing a one-dimensional convolutional layer ($\text{Conv1d}$) sweeping across the temporal grid of the sequence, configured with an output channel size equal to $D$, a kernel size equal to $P$, and a stride parameter equal to $S_{patch}$:

$$Z_{patch} = \text{Conv1d}(X_{seasonal})$$

The physical rationale of this convolutional embedding block is to operate as an adaptive localized band-pass filter, smoothing the raw signal variations and compressing discrete kinematic impulse patterns within a localized window into high-density representation vectors.

##### 2) Channel Independence (CI) Rule
To thoroughly suppress cross-channel noise propagation and prevent collinearity leakage between distinct acceleration measuring axes, the batch dimension ($B$) and sensor channel dimension ($C$) are flattened to transform the latent tensor configuration:

$$Z_{patch} \in \mathbb{R}^{B \times C \times D \times M} \rightarrow Z_{flat} \in \mathbb{R}^{(B \cdot C) \times D \times M}$$

This regularizing constraint forces the subsequent encoder layers to evaluate sequence data from each acceleration sensor as an isolated single-channel entity, preserving the unique physical dynamics of individual mechanical measurement vectors.

##### 3) Hybrid Mamba-CNN Block
The channel-independent latent representation $Z_{flat}$ is subsequently forwarded through a series of hybrid selective state space blocks integrated with convolutional layers. Within each block, the hidden projection signal $H_{in}$ at patch step $m$ is initially routed to a localized 1D convolutional branch with a kernel width $K$ and a non-linear SiLU activation function to eliminate high-frequency instrument noise:

$$H_{conv} = \text{SiLU}(\text{Conv1d}(H_{in}))$$

The filtered feature stream $H_{conv}$ serves as the input driver for the selective continuous state space model ($\text{SSM}$), which executes hidden state transitions using data-dependent coefficient matrices:

$$h_t = A h_{t-1} + B H_{conv}[t]$$

$$H_{mamba} = C h_t + D H_{conv}[t]$$

The embedding of a 1D CNN layer directly prior to the linear scanning mechanism of Mamba structures a robust localized noise barrier, preventing the saturation of hidden recurrent states when subjected to large background white noise fields.

##### 4) Seasonal Forecasting Head
The global temporal context vector generated by the Mamba layers, denoted as $Z_{global}$, is passed to a dedicated forecasting head that projects the latent representation back into the discrete time domain of the future forecast horizon:

$$\hat{Y}_{seasonal} = W_{forecasting} \cdot Z_{global} + b_{forecasting}$$

#### F. Physics-Informed Statistical Head (Stats Head)

> 📌 **[ĐỀ XUẤT SỬA ĐỔI / PROPOSED REVISION - EXTRACTING FROM RAW SIGNAL & LATEX FORMULAS]**
> *(Clarification: Specified that the Stats Head extracts features from raw un-filtered signals to preserve low-frequency descriptors like Mean and avoid filtering bias. Completed all equations in standard LaTeX syntax).*
> 
> To supplement purely data-driven, black-box hidden spaces that operate in a structurally uninterpretable manner, a physical statistical head is structured to anchor the latent optimization within explicit mechanical descriptors. This block extracts an 8-dimensional time-domain statistical vector directly from the **raw unfiltered historical lookback window $X_{raw} \in \mathbb{R}^{C \times L}$** utilizing established mechanical engineering formulations:
> 
> * **Mean** (Indicates structural eccentricity and DC-offset components):
>   $$\mu = \frac{1}{L} \sum_{i=1}^L X_{raw}[i]$$
> * **Standard Deviation** (Measures the amplitude of energy variance surrounding the mean tracking profile):
>   $$\sigma = \sqrt{\frac{1}{L} \sum_{i=1}^L (X_{raw}[i] - \mu)^2}$$
> * **Root Mean Square (RMS)** (Acts as the primary metric tracking total destructive structural fatigue energy):
>   $$\text{RMS} = \sqrt{\frac{1}{L} \sum_{i=1}^L (X_{raw}[i])^2}$$
> * **Peak-to-Peak** (Captures the absolute maximum shock amplitude range within the window):
>   $$X_{p-p} = \max(X_{raw}) - \min(X_{raw})$$
> * **Skewness** (Measures distribution asymmetry, sensitive to initial surface pitting asymmetry):
>   $$\text{Skew} = \frac{1}{L \cdot \sigma^3} \sum_{i=1}^L (X_{raw}[i] - \mu)^3$$
> * **Kurtosis** (Provides maximum statistical sensitivity to transient impact shocks during initial micro-cracking stages):
>   $$\text{Kurt} = \frac{1}{L \cdot \sigma^4} \sum_{i=1}^L (X_{raw}[i] - \mu)^4$$
> * **Crest Factor** (The ratio of peak amplitude to the RMS baseline, isolating localized cracking pulses from background energy):
>   $$\text{CF} = \frac{\max(|X_{raw}|)}{\text{RMS}}$$
> * **Shape Factor** (The ratio of RMS to the mean absolute value, reflecting global structural changes in the wave profile):
>   $$\text{SF} = \frac{\text{RMS}}{\frac{1}{L} \sum_{i=1}^L |X_{raw}[i]|}$$
> 
> Vectơ đặc trưng vật lý 8 chiều $V_{stats} \in \mathbb{R}^{C \times 8}$ này được chuẩn hóa qua tầng BatchNormalization và chuyển đổi tuyến tính trước khi thực hiện phép nối (concatenate) trực tiếp với vectơ không gian ẩn của luồng Mamba:
> 
> $$Z_{fused} = \text{Concat}(Z_{global}, \text{Linear}(V_{stats}))$$
> 
> This architectural loop maintains a modular design controlled via the system configuration parameter $\text{use\_stats}$. If $\text{use\_stats} = \text{False}$ is specified, the statistical extraction pathway is automatically bypassed to return a pure data-driven seasonal Mamba forecast.

#### G. Learnable Dual-Stream Mixing Module
To bypass the strict limitations of direct element-wise integration which enforces fixed, non-adjustable contribution ratios between components, the proposed framework implements a learnable dual-stream mixing module governed by a dynamic weighting parameter $\alpha_c$ optimized independently per sensor channel $c$:

$$\hat{Y}_c = \sigma(\alpha_c) \hat{Y}_{trend, c} + (1 - \sigma(\alpha_c)) \hat{Y}_{seasonal, c}$$

where $\sigma$ represents the standard mathematical Sigmoid activation function constraining the parameter bounds to $(0, 1)$, and $\alpha_c$ denotes a learnable weight parameter optimized via backpropagation gradient descents corresponding to each physical sensor location.

This learnable mixing paradigm empowers the network to automatically adjust the relative dominance between the slow-moving physical degradation pathway (the trend component) and rapid non-linear transient vibrations (the seasonal component) optimized for specific sensor mounting points, thereby maximizing the reconstruction accuracy of the synthesized forecast sequence $\hat{Y}$.

#### H. Anomaly Scoring and Leakage-Free POT Thresholding
Following the acquisition of the composite forecast sequence $\hat{Y}_t$, the Anomaly Score at each time window $t$ is formally defined via the Mean Squared Error (MSE) computed across all sensor dimensions and steps within the horizon $H$:

$$S_t = \frac{1}{C \cdot H} \sum_{c=1}^C \sum_{h=1}^H (Y_{t, c, h} - \hat{Y}_{t, c, h})^2$$

To establish dynamic decision boundaries online in an objective manner, Extreme Value Theory (EVT) utilizing the Peak Over Threshold (POT) technique is integrated directly into the inference loop. The calibration workflow strictly honors a leakage-free constraint by executing parameter estimation over the error distribution derived exclusively from the historical early-stage healthy operational sequences, ensuring absolute isolation from any future degradation or target fault profiles. The execution procedure is structured via discrete mathematical phases:

1. An initial baseline anchor threshold $th_0$ is established by computing a high-percentile marker (e.g., $98\%$) from the anomaly score sequence generated across the healthy training baseline.
2. The extreme positive excesses exceeding the anchor threshold are filtered and aggregated: $E_t = \{S_t - th_0 \mid S_t > th_0\}$.
3. The set of excess values $E_t$ is fitted to a Generalized Pareto Distribution (GPD) to mathematically bound the asymptotic behavior of the distribution tail:
   $$F(x) = 1 - \left(1 + \frac{\gamma x}{\beta}\right)^{-\frac{1}{\gamma}}$$
   where $\gamma$ and $\beta$ denote the shape and scale parameters, respectively, optimized via the Maximum Likelihood Estimation (MLE) method.
4. The final dynamic decision boundary $z_q$ corresponding to a configured, conservative target alarm probability $q$ (e.g., $q = 10^{-4}$) is computed as follows:
   $$z_q = th_0 + \frac{\beta}{\gamma} \left( \left( \frac{n \cdot q}{N_z} \right)^{-\gamma} - 1 \right)$$
   where $n$ represents the total volume of baseline observation samples, and $N_z$ is the historical volume of excess samples violating the initial anchor threshold $th_0$.

During the real-time online monitoring stage, any testing window exhibiting an anomaly score satisfying the logical condition $S_t > z_q$ is immediately flagged as a structural anomaly instance, ensuring a highly practical implementation that systematically eliminates false alarms caused by non-stationary operational variations.
