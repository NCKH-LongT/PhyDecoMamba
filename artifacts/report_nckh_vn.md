Hybrid Mamba-CNN Architecture with Physics-Informed Stats Head for Leakage-Free Bearing Anomaly Detection

Truong Binh Thuan, Ly Hung Lam

Faculty of Information Technology

Ton Duc Thang University

Ho Chi Minh City, Vietnam

e-mail: sunbv56@gmail.com, lyhunglam2004@gmail.com

Truong Long

Falcuty of Software of Engineering

FPT University Ho Chi Minh City

Ho Chi Minh City, Vietnam

e-mail: longt5@fpt.edu.vn

Abstract: Một kiến trúc lai Mamba-CNN tích hợp khối thống kê dẫn đường bằng vật lý được đề xuất cho bài toán phát hiện bất thường vòng bi trong điều kiện vận hành không dừng. Các mạng chuỗi học sâu truyền thống thường gặp khó khăn trong việc cô lập các đặc trưng suy thoái cấu trúc do ảnh hưởng từ nhiễu nền công nghiệp nặng và các nút thắt cổ chai tính toán trên các quỹ đạo thời gian chu kỳ dài. Để giải quyết những hạn chế này, một khung mô hình thích ứng phân phối sử dụng phương pháp phân tách chuỗi thời gian được giới thiệu nhằm cô lập các thành phần xu hướng và mùa vụ từ tín hiệu rung động thô. Các biểu diễn thời gian này sau đó được làm giàu bởi một lớp đặc trưng thống kê vật lý tám chiều được trích xuất từ cửa sổ quan sát quá khứ, tích hợp các chỉ số chẩn đoán cốt lõi bao gồm giá trị hiệu dụng (RMS) và độ nhọn (Kurtosis) nhằm đảm bảo tính diễn giải cơ học. Các luồng đặc trưng tích hợp được xử lý thông qua bộ mã hóa Mamba độc lập kênh để mô hình hóa hiệu quả các phụ thuộc thời gian chu kỳ dài với độ phức tạp tuyến tính. Quá trình tái tạo tín hiệu được thực hiện thông qua các quy trình dự báo và khử chuẩn hóa tuần tự, trong đó sai số bình phương trung bình (MSE) được sử dụng làm điểm số bất thường. Nhằm đảm bảo quy trình hiệu chuẩn không rò rỉ dữ liệu đồng bộ với việc triển khai công nghiệp thực tế, các ngưỡng quyết định cục bộ được thiết lập cho từng vòng bi bằng kỹ thuật vượt ngưỡng (POT) áp dụng nghiêm ngặt trên các phân đoạn dữ liệu khỏe mạnh giai đoạn đầu. Trong các thiết lập được đánh giá, các mô hình đối chứng—bao gồm mạng bộ nhớ ngắn dài hạn (LSTM), ModernTCN, PatchTST và SimpleMamba—được tự động điều chỉnh quy mô nhằm cân bằng ngân sách tham số. Các kết quả thực nghiệm chỉ ra rằng hiệu suất cấu trúc được nâng cao, độ chính xác phát hiện được cải thiện và tỷ lệ báo động giả được giảm thiểu trên các tập dữ liệu rung động thế giới thực.

Keywords: mô hình không gian trạng thái chọn lọc, kiến trúc lai Mamba-CNN, phân tách chuỗi, phát hiện bất thường vòng bi, hiệu chuẩn không rò rỉ dữ liệu, kỹ thuật vượt ngưỡng (Peak Over Threshold).

Introduction

Độ tin cậy và an toàn vận hành trong quản lý tài sản công nghiệp hiện đại phụ thuộc lớn vào quá trình giám sát sức khỏe liên tục của các hệ thống máy móc quay [14]. Trong các hệ thống cơ khí này, vòng bi được phân loại là cấu phần cốt lõi, do sự suy thoái cấu trúc của chúng trực tiếp dẫn đến những hư hỏng hệ thống nghiêm trọng và tổn thất kinh tế lớn [15]. Do đó, phát hiện bất bất thường không giám sát dựa trên tín hiệu rung động đã trở thành nền tảng của các khung kiến trúc Quản lý Sức khỏe và Dự đoán (PHM). Tín hiệu rung động thô thu thập từ môi trường vận hành thực tế thường biểu hiện các phân phối không dừng, động học phi tuyến tính phức tạp và liên tục bị biến dạng bởi nhiễu nền công nghiệp nặng. Mặc dù các kỹ thuật xử lý tín hiệu truyền thống cung cấp các năng lực chẩn đoán cơ bản, các xung va đập chuyển tiếp và tín hiệu âm thanh tinh vi phát sinh trong giai đoạn chớm lỗi thường bị che lấp bởi nhiễu vận hành, đặt ra yêu cầu về các mô hình chuỗi tiên tiến có khả năng cô lập các hành vi bất thường với độ trung thực cấu trúc cao.

Trong những năm gần đây, các phương pháp học sâu đã được áp dụng để tự động hóa quá trình trích xuất đặc trưng chuỗi thời gian và phân loại trạng thái sức khỏe máy móc [14]. Mạng hồi quy và mạng tích chập thời gian đã được triển khai để theo dõi chuỗi, song các nút thắt cổ chai cập nhật lặp và trường đón nhận cố định đã hạn chế hiệu quả của chúng trên các quỹ đạo thời gian dài. Nhằm bắt giữ các tương tác chu kỳ dài, các kiến trúc tiên tiến (SOTA)—bao gồm Anomaly Transformer [5] và TimesNet [6]—đã khai thác cơ chế tự chú ý và phân tích đa tần số. Tuy nhiên, các khung mô hình này phải gánh chịu độ phức tạp tính toán bậc hai () và dễ rơi vào hiện tượng quá khớp khi đối mặt với nhiễu công nghiệp. Hơn nữa, các mô hình đối chứng nâng cao khác như ModernTCN [2] và PatchTST [7] thực hiện phân đoạn tín hiệu thành các chuỗi con hoặc mảnh cục bộ nhằm tối ưu hóa hiệu suất xử lý. Bất chấp những cải tiến cấu trúc này, các phương pháp trên vận hành chủ yếu dưới ràng buộc độc lập kênh, dẫn đến sự thất bại trong việc dung hợp hiệu quả mối tương quan xuyên kênh với ngữ cảnh thời gian toàn cục, gây ra tỷ lệ báo động giả tăng cao trong các hệ thống công nghiệp đa cảm biến.

Gần đây hơn, các mô hình không gian trạng thái chọn lọc, đặc biệt là kiến trúc Mamba [16], đã được giới thiệu để thiết lập độ phức tạp thời gian tuyến tính () trong khi vẫn duy trì cơ chế quét chọn lọc mạnh mẽ. Các biến thể chuyên biệt—bao gồm FEMamba [8] và TFG-Mamba [9]—đã cố gắng tích hợp các đặc trưng miền thời-tần để nâng cao độ chính xác chẩn đoán. Tuy nhiên, một khoảng trống nghiên cứu cốt lõi vẫn chưa được giải quyết trong các khung mô hình SOTA này. Thứ nhất, các mạng chẩn đoán dựa trên Mamba hiện tại coi quá trình dự báo tín hiệu và thiết lập ngưỡng là các quy trình tách rời, bỏ qua sự dịch chuyển phân phối vốn có của tín hiệu không dừng. Thứ hai, các mạng chuỗi thuần túy hướng dữ liệu vận hành như các hộp đen không thể diễn giải, tách rời điểm số bất thường thống kê khỏi bản chất vật lý suy thoái cơ học. Thứ ba, các cơ chế thiết lập ngưỡng hiện tại thường xuyên gây ra hiện tượng rò rỉ dữ liệu khi sử dụng dữ liệu kiểm định lỗi toàn cục hoặc tương lai trong pha hiệu chuẩn, khiến chúng mất đi tính thực tế trong các quy trình triển khai trực tuyến vốn chỉ tiếp cận được các đường cơ sở khỏe mạnh cục bộ trong lịch sử.

Để giải quyết các thách thức liên đới này, một hệ thống dự báo Mamba-CNN lai phân rã chuỗi dẫn đường bằng vật lý được đề xuất trong nghiên cứu này cho bài toán phát hiện bất thường vòng bi không rò rỉ dữ liệu. Kiến trúc cốt lõi tích hợp một mô-đun phân tách chuỗi thời gian với bộ lọc trung bình trượt để cô lập các xu hướng suy thoái tần số thấp khỏi các thành phần xung đột biến mùa vụ tần số cao. Các luồng đặc trưng mùa vụ sau đó được xử lý thông qua một bộ mã hóa Mamba độc lập kênh nhằm tối đa hóa khả năng trích xuất phụ thuộc toàn cục mà không làm lan truyền nhiễu xuyên kênh. Để tích hợp tri thức miền kỹ thuật và đảm bảo tính diễn giải cơ học, các biểu diễn ẩn được tăngروع bởi một lớp đặc trưng thống kê vật lý tám chiều (Stats Head) được trích xuất trực tiếp từ cửa sổ quan sát quá khứ. Lớp đặc trưng này bao gồm: Giá trị trung bình, Độ lệch chuẩn, Giá trị hiệu dụng, Biên độ đỉnh-đỉnh, Độ lệch, Độ nhọn, Hệ số đỉnh và Hệ số hình dạng, dung hợp các biểu diễn hướng dữ liệu thô với các đặc tính cơ học đã được thiết lập trong một đầu dự báo dung hợp chuyên dụng.

Các đóng góp chính của nghiên cứu này được tóm tắt như sau. Thứ nhất, một mô hình dự báo lai Mamba-CNN với cơ chế phân rã chuỗi được giới thiệu, đạt độ phức tạp tuyến tính đồng thời loại bỏ nhiễu xuyên kênh. Thứ hai, một khối thống kê dẫn đường bằng vật lý được cấu trúc nhằm thu hẹp khoảng cách giữa không gian ẩn hướng dữ liệu và các bộ mô tả cơ học, đảm bảo tính diễn giải cấu trúc. Thứ ba, một quy trình hiệu chuẩn không rò rỉ thông tin sử dụng Lý thuyết giá trị cực biên thông qua kỹ thuật vượt ngưỡng (POT) được thực hiện, trong đó các ranh giới động được tính toán nghiêm ngặt và duy nhất từ các phân đoạn dữ liệu khỏe mạnh ở giai đoạn đầu. Phần còn lại của bài báo được tổ chức như sau: Phần II tổng quan các tài liệu liên quan và các mô hình chuỗi đối chứng. Phần III chi tiết hóa cấu trúc toán học của phương pháp đề xuất. Phần IV trình bày thiết lập thực nghiệm, đồng bộ ngân sách tham số baseline và phân tích hiệu năng so sánh. Phần V kết luận nghiên cứu và thảo luận về các định hướng phát triển tương lai.

Related work

Mô hình hóa chuỗi học sâu trong Quản lý Sức khỏe và Dự đoán (Prognostics and Health Management — PHM)

Các phân tích thống kê thực nghiệm chỉ ra rằng các sự cố liên quan đến vòng bi chiếm khoảng 40%–50% tổng số hư hại cấu trúc của các hệ thống máy móc quay, trực tiếp gây ra hiện tượng dừng máy ngoài kế hoạch và tổn thất kinh tế nghiêm trọng [15]. Nhằm trích xuất các đặc trưng miền thời-tần một cách không giám sát từ tín hiệu rung động phi tuyến, các khung tối ưu hóa bộ lọc Wavelet học được (learnable Wavelet)—chẳng hạn như kiến trúc DeSpaWN [1]—đã chứng minh hiệu quả xử lý tín hiệu nền tảng vượt trội. Tuy nhiên, các đặc trưng tần số tĩnh này thường bỏ sót các chuyển dịch động học dài hạn khi cấu trúc cơ học suy thoái theo thời gian, đặt ra nhu cầu về các mô hình có khả năng học biểu diễn chuỗi thời gian dài [14].

Để bắt giữ các phụ thuộc thời gian này, các kiến trúc học sâu như mạng bộ nhớ ngắn dài hạn (Long Short-Term Memory — LSTM) và mạng tích chập thời gian (Temporal Convolutional Network — TCN) đã được áp dụng rộng rãi. Mặc dù vậy, LSTM phụ thuộc vào cơ chế cập nhật trạng thái ẩn tuần tự, tạo ra các nút thắt cổ chai tính toán nghiêm trọng làm ngăn cản khả năng tính toán song song và gây ra hiện tượng tiêu biến đạo hàm trên các cửa sổ dài. Trong khi đó, TCN giải quyết khả năng tính toán song song thông qua các bộ lọc giãn, song trường đón nhận vẫn bị giới hạn bởi cấu hình kích thước bộ lọc cố định, dẫn đến sự thất bại trong việc bao quát toàn bộ chu kỳ quay của vòng bi dưới điều kiện tần số lấy mẫu của cảm biến quá cao.

Đối với tác vụ phát hiện bất thường không giám sát, các mạng tự mã hóa (Autoencoders — AE) truyền thống thường được sử dụng để học biểu diễn trạng thái khỏe mạnh và phát hiện lỗi dựa trên sai số tái cấu trúc. Tuy nhiên, các mô hình này thực tế thường gánh chịu hiện tượng "khái quát hóa quá mức" (over-generalization), nghĩa là không gian ẩn vô tình tái cấu trúc chuẩn xác cả những vi xung động chớm lỗi, dẫn đến việc triệt tiêu sai số tái cấu trúc và làm tăng tỷ lệ sót lỗi nghiêm trọng [4]. Nhằm khắc phục hạn chế này, các nghiên cứu gần đây như TimeRCD [4] đề xuất cơ chế học sự khác biệt ngữ nghĩa liên hợp, hoặc xu hướng chuyển dịch hẳn từ mạng tự mã hóa sang bài toán dự báo chuỗi thời gian của cửa sổ kế tiếp thay vì tái cấu trúc thô. Song song đó, các nghiên cứu phản biện như ModernTCN Revisited [3] đã chỉ ra rằng việc thiết lập giao thức thực nghiệm không chặt chẽ dễ dẫn đến rò rỉ thông tin kiểm định. Do đó, nghiên cứu này kế thừa tư duy dự báo chuỗi thời gian nhằm triệt tiêu hiện tượng khái quát hóa quá mức, đồng thời áp dụng giao thức đồng bộ hóa quy mô tham số nghiêm ngặt để đảm bảo quá trình so sánh baseline đạt tính công bằng tuyệt đối.

Mô hình chuỗi thời gian dựa trên cơ chế Chú ý (Transformer)

Sự phát triển của cơ chế tự chú ý (Self-Attention) trong các cấu trúc Transformer đã nâng cao năng lực bắt giữ các mối tương quan dài hạn của chuỗi thời gian. Trong bài toán phát hiện bất thường, kiến trúc Anomaly Transformer [5] đã đặt nền móng bằng cách đo lường sự sai khác liên hợp giữa phân phối chú ý cục bộ và toàn cục để cô lập các điểm dị thường. Nhằm giải quyết triệt để đặc tính không dừng của tín hiệu công nghiệp, mô hình TimesNet [6] đã biến đổi chuỗi thời gian một chiều (1D) thành Tensor hai chiều (2D) thông qua phân tích đa chu kỳ Fourier, từ đó áp dụng các phép toán tích chập 2D trên các tensor chu kỳ được gấp cuộn (2D convolutional operations on folded period tensors) để khai thác đồng thời các biến động nội chu kỳ và liên chu kỳ. Tuy nhiên, việc áp dụng các phép toán tích chập dựa trên mạng tích chập (Convolutional Neural Network — CNN) 2D liên tục trên các Tensor tần số lớn tạo ra chi phí tính toán rất cao, đồng thời cấu trúc CNN thuần túy của TimesNet thiếu khả năng quét chọn lọc các chuỗi phụ thuộc dài hạn [6].

Để tối ưu hóa hiệu suất, mô hình PatchTST [7] giới thiệu cơ chế vá mảnh thời gian (patching) kết hợp ràng buộc độc lập kênh (Channel Independence — CI), giúp giảm chiều dài chuỗi đầu vào của mạng Attention. Dù vậy, đối với các cửa sổ lịch sử siêu dài—cửa sổ quan sát lịch sử (lookback window) được mở rộng tối đa để bao quát các tần số quay thấp trong công nghiệp—độ phức tạp tính toán của cơ chế tự chú ý trong các cấu trúc Transformer nói chung vẫn tăng theo hàm bậc hai đối với số lượng mảnh thời gian (patches), gây quá tải bộ nhớ phần cứng nghiêm trọng [7]. Hơn nữa, các bộ mã hóa Transformer nguyên bản thiếu các thiên kiến quy nạp tịnh tiến cục bộ. Khi tín hiệu rung động bị lẫn nhiễu nền công nghiệp nặng, Transformer dễ bị đánh lừa bởi các đỉnh nhiễu ngẫu nhiên và gán nhãn chúng thành các dấu hiệu suy thoái cấu trúc, gây ra tỷ lệ báo động giả tăng cao [14]. Kiến trúc lai Mamba-CNN đề xuất trong nghiên cứu này khắc phục hạn chế trên bằng cách tích hợp bộ quét chọn lọc tuyến tính của Mamba để xử lý đặc trưng toàn cục và các nhánh tích chập cục bộ để triệt tiêu nhiễu tần số cao.

Mô hình Không gian Trạng thái chọn lọc (Mamba) trong miền PHM

Để phá vỡ giới hạn tính toán bậc hai của Transformer trên các chuỗi dài, mô hình không gian trạng thái chọn lọc (Mamba) [16] đã thiết lập một cột mốc mới, đạt độ phức tạp thời gian tuyến tính () nhờ cơ chế quét chọn lọc phụ thuộc dữ liệu đầu vào. Trong miền PHM và chẩn đoán lỗi vòng bi, các biến thể dựa trên Mamba đã xuất hiện mạnh mẽ trong giai đoạn 2025–2026. Kiến trúc FEMamba [8] tích hợp khối trích xuất đặc trưng thích ứng (Multi-scale Feature Extraction Slot — MFES) và mô-đun ràng buộc không gian suy thoái (Degradation Space Guided Regression — DSGR) phục vụ cho bài toán hồi quy dự đoán tuổi thọ còn lại (Remaining Useful Life — RUL). Khung mô hình TFG-Mamba [9] thực hiện dung hợp miền thời-tần thông qua cơ chế cổng Mamba điều hướng để bắt giữ xu hướng suy thoái dài hạn của máy móc quay.

Đối với các cấu trúc đa biến phức tạp, mô hình TimeMachine [10] triển khai kiến trúc Quadruple-Mamba nhằm tích hợp đồng thời cơ chế trộn kênh và độc lập kênh, trong khi kiến trúc TSC-Mamba [11] áp dụng quy trình "Phân tách—Lan truyền—Tương quan chéo" qua mô-đun tương quan dung hợp thông tin kênh (Channel Information Fusion Module — CIFM) bậc thấp để tối ưu hóa biểu diễn chuỗi. Nhằm hướng tới việc triển khai thực tế, mô hình đồ thị phân tách tuyến tính (Linear Decoupled Graph Model — LDGM) [12] hỗ trợ phân tách đặc trưng động học toàn cục và vi xung động cục bộ thông qua cơ chế Mamba đa góc nhìn (Multi-Perspective Mamba — Mamba-MP) tại thiết bị biên, và nghiên cứu PG-TMT [13] kết hợp mô hình lai Tiny-Mamba Transformer dẫn đường bằng vật lý (Physics-Guided Tiny-Mamba Transformer — PG-TMT) với tri thức dải tần vật lý.

Mặt khác, các kỹ thuật phân tách tín hiệu chuỗi thời gian—chẳng hạn như Phân tách Xu hướng—Mùa vụ dựa trên LOESS (Seasonal-Trend Decomposition using LOESS — STL) và Phân tích Phổ Đơn kênh (Singular Spectrum Analysis — SSA)—đã được thiết lập rộng rãi trong giám sát sức khỏe máy móc nhằm cô lập các đặc tính suy thoái dài hạn khỏi các biến động vận hành phức tạp. STL trích xuất hiệu quả xu hướng suy thoái đơn điệu đại diện cho sự mài mòn vật lý trong khi loại bỏ các bất thường có chu kỳ vận hành. Tương tự, SSA tạo điều kiện phân tách không mô hình (model-free) các quỹ đạo suy thoái mịn khỏi nhiễu trắng ngẫu nhiên. Gần đây, các khung lai kết hợp phân tách chế độ—ví dụ: Phân tách Chế độ Biến phân (Variational Mode Decomposition — VMD) hoặc Phân tách Chế độ Kinh nghiệm (Empirical Mode Decomposition — EMD)—với mô hình không gian trạng thái chọn lọc như kiến trúc MD-BiMamba [21] đã chứng minh rằng việc phân tách các tín hiệu rung động thô trước khi mã hóa Mamba giúp giảm thiểu đáng kể hiệu ứng che lấp lỗi (fault masking), cho phép cơ chế quét chọn lọc tập trung chặt chẽ vào các xung động do lỗi gây ra.

Tuy nhiên, một giới hạn cốt lõi của các nghiên cứu tiên tiến nhất (State-of-the-Art — SOTA) dựa trên Mamba này là chúng hầu hết được cấu trúc như các mô hình học máy có giám sát hoặc hồi quy tuổi thọ yêu cầu nhãn lỗi đầy đủ. Trong kịch bản công nghiệp thực tế, dữ liệu nhãn lỗi cực kỳ khan hiếm và đắt đỏ, đòi hỏi các giải pháp phát hiện bất thường không giám sát [14], [15]. Hơn thế nữa, việc sử dụng Mamba thuần túy hướng dữ liệu đóng vai trò như một hộp đen thiếu tính diễn giải cơ học đối với kỹ sư vận hành [13]. Mamba quét tuần tự tín hiệu thời gian nhưng không có cơ chế trực tiếp tích hợp các chỉ số cơ học đặc trưng của vòng bi. Nghiên cứu này giải quyết khoảng trống học thuật bằng cách nhúng trực tiếp một khối Stats Head vật lý 8 chiều vào không gian ẩn của Mamba, hướng dẫn quá trình dự báo không giám sát thông qua các bộ mô tả cơ học tường minh, đồng thời tối ưu hóa Mamba trên cửa sổ quan sát siêu dài để bắt trọn các chu kỳ suy thoái tần số thấp với chi phí tính toán tuyến tính.

Phương pháp thiết lập ngưỡng động và hiệu chuẩn phát hiện bất thường

Một hệ thống phát hiện bất thường hoàn chỉnh không chỉ phụ thuộc vào kiến trúc mạng mà còn bị quyết định bởi cơ chế hiệu chuẩn ngưỡng quyết định. Các ngưỡng tĩnh (3-Sigma, IQR) hoàn toàn thất bại dưới điều kiện vận hành thay đổi do sự dịch chuyển phân phối của dữ liệu miền đích. Ngược lại, các ngưỡng động dựa trên mô hình thống kê như mô hình hỗn hợp Gauss (Gaussian Mixture Models — GMM) lại thiếu cơ sở toán học vững chắc khi xử lý các giá trị cực biên nằm ngoài phân phối huấn luyện. Việc tích hợp Lý thuyết giá trị cực biên (Extreme Value Theory — EVT) qua kỹ thuật vượt ngưỡng (Peak Over Threshold — POT) [17] đã chứng minh năng lực thiết lập ranh giới quyết định tập mở chính xác dựa trên phân phối Pareto tổng quát (Generalized Pareto Distribution — GPD).

Mặc dù vậy, các nghiên cứu PHM hiện tại ứng dụng POT thường mắc phải lỗi rò rỉ dữ liệu tương lai. Cụ thể, các nghiên cứu tiêu biểu dựa trên mô hình đồ thị ngẫu nhiên chuỗi thời gian (Stochastic Recurrent Neural Network — Stochastic RNN) như OmniAnomaly [18], mạng tự mã hóa đối kháng như USAD [19], hay Transformer đối kháng như TranAD [20] đều sử dụng toàn bộ tập kiểm định—bao gồm cả các phân đoạn đã suy thoái sâu hoặc lỗi hoàn toàn ở tương lai—để tối ưu hóa các tham số ngưỡng nhằm đạt điểm F1-score cao nhất trên giấy tờ. Điều này vô tình biến quá trình thiết lập ngưỡng thành một tác vụ học nửa giám sát, không thể triển khai trực tuyến trên thiết bị thực tế nơi hệ thống hoàn toàn không có cơ hội tiếp cận trước các phân phối biên dị thường.

Để giải quyết triệt để vấn đề này, mô hình đề xuất thực thi quy trình hiệu chuẩn cục bộ nghiêm ngặt. Việc tính toán và khớp tham số POT được thực hiện duy nhất trên phân đoạn dữ liệu khỏe mạnh ban đầu (condition variable == 0) mà không sử dụng bất kỳ thông tin lỗi nào từ tương lai. Dưới điều kiện vận hành thay đổi, mô hình dự báo lai Mamba-CNN sẽ tự động thích ứng với các phân phối động của tín hiệu khỏe mạnh [15], giữ cho sai số dự báo luôn nằm dưới ngưỡng cực biên đã được hiệu chuẩn tĩnh ban đầu, đảm bảo tính thực tiễn và khả năng triển khai trực tuyến tuyệt đối.

Bảng Ma trận So sánh Tài liệu (Literature Matrix Table)

TABLE I COMPREHENSIVE LITERATURE MATRIX AND STRUCTURAL COMPARISON WITH THE PROPOSED PARADIGM

| Nghiên cứu (Study) | Tác vụ (Task) | Tập dữ liệu (Dataset) | Phương pháp (Method) | Điểm mạnh (Strength) | Hạn chế (Limitation) | Mối liên hệ với nghiên cứu này (Relation to this work) |
| --- | --- | --- | --- | --- | --- | --- |
| OmniAnomaly (2019) [18] | Phát hiện bất thường không giám sát | Chuỗi thời gian đa biến công nghiệp | Mô hình đồ thị có hướng tái cấu trúc ngẫu nhiên (Stochastic RNN). | Khai thác xác suất tái cấu trúc để biểu diễn các biến động động học ngẫu nhiên. | Gánh chịu hiện tượng khái quát hóa quá mức (over-generalization) làm sót lỗi sớm; hiệu chuẩn ngưỡng gây rò rỉ dữ liệu tương lai. | Thay đổi từ khung tái cấu trúc sang khung dự báo chuỗi thời gian, áp dụng POT cục bộ không rò rỉ thông tin. |
| USAD (2020) [19] | Phát hiện bất thường không giám sát | Tập dữ liệu kiểm thử chuẩn (SWaT, WADI) | Mạng tự mã hóa huấn luyện đối kháng (Adversarially trained autoencoders). | Tăng cường độ ổn định và tốc độ hội tụ của cấu trúc Autoencoder thông qua kiến trúc học đối kháng. | Vận hành như một hộp đen thuần túy; tối ưu hóa ngưỡng quyết định dựa trên toàn bộ phân phối tập validation. | Tích hợp lớp Stats Head vật lý 8 chiều để phá vỡ tính chất hộp đen và cô lập pha hiệu chuẩn ngưỡng POT ở giai đoạn đầu khỏe mạnh. |
| TranAD (2022) [20] | Phát hiện bất thường không giám sát | Chuỗi thời gian đa biến phức tạp | Transformer kết hợp học đối kháng và cơ chế tự điều kiện hóa (Self-conditioning). | Đạt tốc độ suy luận nhanh và độ chính xác cao nhờ các thuộc tính đối kháng của Transformer. | Chi phí bộ nhớ lớn trên chuỗi siêu dài; tối ưu hóa siêu tham số ngưỡng POT phụ thuộc vào dữ liệu lỗi tương lai. | Ứng dụng cấu trúc lai Mamba-CNN với chi phí tuyến tính và thực thi nghiêm ngặt điều kiện hiệu chuẩn tĩnh (condition == 0). |
| Anomaly Transformer (2022) [5] | Phát hiện bất thường không giám sát | Chuỗi thời gian đa miền tổng quát (SMD, MSD, v.v.) | Cơ chế tính toán Sự khác biệt liên hợp (Association Discrepancy) dựa trên Transformer. | Cô lập hiệu quả các dị thường chu kỳ nhờ phân phối chú ý cục bộ và toàn cục. | Độ phức tạp tính toán bậc hai (); cực kỳ nhạy cảm với nhiễu nền công nghiệp. | Mô hình đề xuất thay thế khối tự chú ý bằng bộ mã hóa Mamba để đạt độ phức tạp tuyến tính . |
| TimesNet (2023) [6] | Dự báo và phát hiện bất thường | Chuỗi thời gian không dừng (Weather, ETT, v.v.) | Biến đổi chuỗi 1D thành cấu trúc 2D thông qua phân tích đa chu kỳ Fourier (FFT) và Inception blocks. | Bắt giữ tốt các đặc tính không dừng nhờ phân tích đa chu kỳ đồng thời. | Chi phí tính toán lớn do xử lý ảnh 2D; cấu trúc CNN thuần túy thiếu khả năng quét chọn lọc chuỗi siêu dài. | Mô hình đề xuất sử dụng cơ chế phân tách chuỗi (Series Decomposition) trực tiếp trên miền thời gian để giảm chi phí FFT. |
| PatchTST (2023) [7] | Dự báo chuỗi thời gian dài | Tập dữ liệu chuỗi thời gian chuẩn (Electricity, Traffic) | Kỹ thuật vá mảnh thời gian (Patching) kết hợp ràng buộc độc lập kênh (Channel Independence). | Giảm độ phức tạp của Transformer xuống bậc tuyến tính; loại bỏ nhiễu xuyên kênh. | Hoàn toàn bỏ sót việc dung hợp tương quan giữa các kênh, làm giảm độ chính xác trong hệ thống đa cảm biến. | Mô hình đề xuất giữ cơ chế độc lập kênh ở nhánh Mamba nhưng dung hợp xuyên kênh ở đầu phát hiện (Fusion Head). |
| MD-BiMamba (2024) [21] | Chẩn đoán lỗi có giám sát (Supervised) | Vòng bi trục động cơ hàng không (Aero-engine bearing) | Kết hợp phân tách chế độ tín hiệu (VMD/EMD) và chiến lược dung hợp đặc trưng hai chiều (Bidirectional Mamba). | Giảm thiểu hiệu ứng che lấp lỗi (fault masking) hiệu quả, tối ưu hóa khả năng quét chọn lọc các xung va đập lỗi. | Vận hành dưới dạng mô hình phân loại có giám sát phụ thuộc vào nhãn lỗi; không có cơ chế thiết lập ngưỡng động trực tuyến chống rò rỉ. | Kế thừa tư duy phân tách tín hiệu chuỗi để làm sạch đặc trưng nhưng phát triển trên khung dự báo không giám sát (unsupervised) phối hợp POT cục bộ. |
| TFG-Mamba (2026) [9] | Dự báo xu hướng suy thoái thiết bị | Tập dữ liệu suy thoái vòng bi thực tế | Dung hợp miền thời-tần thông qua cơ chế cổng Mamba điều hướng (Time-Frequency Gated). | Khai thác sâu các đặc trưng tần số để theo dõi quỹ đạo suy thoái dài hạn. | Vận hành như một hộp đen hướng dữ liệu thuần túy; yêu cầu nhãn giám sát; không có cơ chế thiết lập ngưỡng chống rò rỉ. | Mô hình đề xuất xây dựng trên khung dự báo không giám sát và tích hợp khối POT chống rò rỉ dữ liệu trực tuyến. |
| PG-TMT (2026) [13] | Giám sát sức khỏe vòng bi trực tuyến | Tập dữ liệu run-to-failure của vòng bi công nghiệp | Bộ mã hóa ba nhánh (Tri-branch) kết hợp Tiny-Mamba, tích chập phân tách sâu và tri thức vật lý dải tần. | Kiến trúc gọn nhẹ phù hợp giám sát thời gian thực; tích hợp tri thức tần số vật lý. | Cấu trúc phân nhánh phức tạp; quá trình tính ngưỡng chẩn đoán vẫn phụ thuộc vào phân phối kiểm định toàn cục. | Mô hình đề xuất đơn giản hóa cấu trúc bằng chuỗi đơn phân rã xu hướng/mùa vụ và cô lập hiệu chuẩn tĩnh ở giai đoạn đầu. |
| EVT Open Set (2022) [17] | Chẩn đoán lỗi tập mở (Open Set) | Tập dữ liệu lỗi vòng bi cơ bản (CWRU) | Tích hợp Lý thuyết giá trị cực biên (EVT-POT) trực tiếp vào không gian ẩn học sâu. | Thiết lập biên quyết định toán học chính xác cho các lỗi chưa từng xuất hiện trong tập huấn luyện. | Quá trình hiệu chuẩn ngưỡng POT sử dụng dữ liệu validation toàn cục, gây rò rỉ thông tin tương lai. | Mô hình đề xuất áp dụng POT nhưng thực thi quy trình hiệu chuẩn cục bộ không rò rỉ (condition == 0). |
| Mô hình Đề xuất (Proposed) | Phát hiện bất thường vòng bi không giám sát | Các tập dữ liệu rung động suy thoái thế giới thực | Mô hình lai Mamba-CNN phân rã chuỗi, tích hợp Stats Head 8 chiều và hiệu chuẩn POT cục bộ. | Độ phức tạp tuyến tính ; diễn giải cơ học rõ ràng; quy trình hiệu chuẩn không rò rỉ dữ liệu thực tế. | Hiệu năng phụ thuộc vào chất lượng của phân đoạn dữ liệu khỏe mạnh ở giai đoạn đầu. | (Thiết lập đường cơ sở khoa học mới cho bài toán PHM không rò rỉ dữ liệu). |

Methodology

Phát biểu Bài toán và Khung Phương pháp Tổng quan (Problem Formulation and Framework Overview)

Quá trình giám sát sức khỏe cấu trúc của vòng bi công nghiệp thông qua dữ liệu chuỗi thời gian được định hình dưới kịch bản phát hiện bất thường không giám sát dựa trên cơ chế dự báo cửa sổ kế tiếp. Giả định một chuỗi tín hiệu rung động đa biến thu thập từ hệ thống cảm biến gia tốc gồm  kênh đo (đại diện cho các hướng thu thập dữ liệu cơ học như trục ngang và trục dọc) với chiều dài lịch sử quan sát (lookback window) là . Tensor dữ liệu đầu vào tại thời điểm  được định nghĩa là . Mục tiêu toán học của hệ thống là dự báo chính xác chuỗi tín hiệu tương lai trong một cửa sổ dự báo (forecast horizon) có độ dài , ký hiệu là , nhằm đối chiếu với chuỗi tín hiệu thực tế tương ứng .

Khung phương pháp đề xuất vận hành theo nguyên lý học biểu diễn không giám sát, trong đó toàn bộ quá trình tối ưu hóa mạng chỉ sử dụng dữ liệu thuộc pha hoạt động lành mạnh ban đầu của thiết bị—được xác định bởi các nhãn chỉ thị trạng thái bằng không (). Cơ sở khoa học của phương pháp này dựa trên giả thuyết rằng mô hình sẽ thiết lập một không gian ẩn tối ưu để tái cấu trúc và dự báo các đặc tính động học bình thường của hệ thống cơ khí. Khi vòng bi xuất hiện các vết nứt rỗ, mài mòn hoặc các dạng tổn thất cấu trúc (fault state), sự xuất hiện của các xung va đập chuyển tiếp phi tuyến tính và sự dịch chuyển phân phối biên sẽ phá vỡ tính quy luật của chuỗi dữ liệu. Hệ quả là sai lệch bình phương giữa tín hiệu thực tế  và tín hiệu dự báo  sẽ tăng vọt, cung cấp một đường cơ sở toán học đáng tin cậy để thiết lập điểm số bất thường (anomaly score) và kích hoạt ranh giới cảnh báo trực tuyến mà không phụ thuộc vào nguồn dữ liệu nhãn lỗi khan hiếm.

Tiền xử lý Tín hiệu qua Bộ lọc Thông cao Butterworth Nhân quả (Causal Butterworth High-Pass Filtering)

Để triệt tiêu các thành phần dao động tần số thấp phát sinh từ động cơ nền, nhiễu môi trường hoặc các đặc tính cơ học không liên quan đến sự suy thoái của vòng bi, tín hiệu rung động thô ban đầu được truyền qua một bộ lọc thông cao Butterworth nhân quả bậc . Hàm truyền đạt bình phương biên độ của bộ lọc trong miền tần số liên tục được biểu diễn dưới dạng:

Trong đó  ký hiệu tần số cắt cấu hình, và  là tần số thành phần của tín hiệu. Để triển khai bộ lọc toán học này trực tiếp vào luồng xử lý dữ liệu thời gian thực rời rạc mà không gây rò rỉ thông tin tương lai, phép biến đổi song tuyến tính (bilinear transformation) được áp dụng nhằm chuyển đổi hàm truyền đạt từ miền tần số liên tục sang miền thời gian rời rạc, thiết lập phương trình sai phân hiệu chỉnh có dạng:

Trong đó  và  lần lượt là tín hiệu trước và sau khi lọc tại bước thời gian , trong khi  và  là hệ số bộ lọc được xác định qua tần số lấy mẫu . Nhờ thiết lập nhân quả này, độ trễ pha sinh ra từ bộ lọc sẽ được mô hình học sâu học cách bù đắp một cách tự nhiên trong quá trình tối ưu hóa dự báo. Ý nghĩa vật lý của cấu phần tiền xử lý này là cô lập và bảo toàn các vi xung động va đập biên độ nhỏ ở dải tần số cao—vốn là chỉ thị nhạy cảm nhất đối với hiện tượng bong tróc hoặc rỗ bề mặt thớ cơ học ở giai đoạn chớm lỗi—đồng thời ổn định hóa phân phối dữ liệu đầu vào.

Khối Phân tách Chuỗi Thời gian Thích ứng (Distribution-Adaptive Series Decomposition)

Đồng bộ với nguyên lý tối giản kiến trúc (architectural parsimony) được giới thiệu trong các mô hình phân tách chuỗi thời gian tiên tiến như kiến trúc DMamba [22], khung phương pháp đề xuất thực hiện tách biệt hoàn toàn quy trình xử lý các luồng thành phần mùa vụ (seasonal) và xu hướng (trend). Bản chất của cơ chế này là phân rã chuỗi tín hiệu đã lọc  thành hai thành phần có đặc tính vật lý và thống kê riêng biệt nhằm tối ưu hóa hiệu suất biểu diễn của mạng học sâu. Quy trình phân tách được thực thi thông qua toán tử trung bình trượt một chiều () trượt dọc theo trục thời gian lịch sử:

Trong đó  là kích thước nhân lọc trung bình trượt. Lớp đệm () áp dụng cơ chế lặp biên để đảm bảo tensor xu hướng  duy trì tính đồng nhất về kích thước hình học với chuỗi gốc.

Thành phần xu hướng  đại diện cho đường trung bình dịch chuyển, phản ánh quá trình tiến triển mài mòn cơ học tần số thấp, chuyển động chậm theo thời gian và có độ phức tạp chiều thấp. Ngược lại, thành phần mùa vụ  cô lập các biến động động học phi tuyến tính mạnh, bao gồm các chu kỳ quay đồng bộ của trục máy và các chuỗi xung va đập chuyển tiếp tần số cao gây ra bởi hư tổn cấu trúc vòng bi. Việc tách biệt này ngăn chặn hiện tượng các thành phần xu hướng năng lượng cao làm lu mờ các vi xung đột biến mùa vụ, cho phép các nhánh mạng chuyên biệt tập trung vào các miền đặc trưng tương thích.

Nhánh Dự báo Xu hướng Tuyến tính (Linear Trend Forecasting Stream)

Mặc dù thể hiện xu hướng suy thoái lũy tiến, thành phần  có độ phức tạp chiều rất thấp và quy luật biến thiên mịn. Việc chuyển luồng dữ liệu này qua các bộ mã hóa có tham số phức tạp như mạng attention hoặc quét không gian trạng thái chọn lọc là không cần thiết, dễ dẫn đến hiện tượng quá khớp (overfitting) và bùng nổ chi phí tính toán. Do đó, một nhánh chiếu tuyến tính trực tiếp (linear projection) được cấu trúc để dự báo thành phần xu hướng tương lai  từ cửa sổ lịch sử:

Trong đó  đại diện cho ma trận trọng số chiếu, và  là vector định thiên (bias). Để tối ưu hóa hiệu suất lưu trữ và tăng cường tính ổn định cho các hệ thống có cửa sổ quan sát lịch sử siêu dài, một toán tử giảm mẫu thích ứng () với bước nhảy  được tích hợp để nén chuỗi xu hướng trước khi thực hiện phép nhân ma trận. Ma trận trọng số  được thiết lập dưới ràng buộc chia sẻ trọng số độc lập kênh (channel-independent weight sharing), nghĩa là một ma trận duy nhất được áp dụng đồng nhất cho mọi kênh cảm biến , giúp giảm thiểu đáng kể số lượng tham số cần huấn luyện và bảo toàn xu hướng suy thoái cơ học chung.

Nhánh Dự báo Mùa vụ dựa trên Mamba-CNN Độc lập Kênh (Seasonal Mamba-CNN Forecasting Stream)

Thành phần mùa vụ  chứa đựng các thông tin động học phi tuyến tính phức tạp cấu thành từ nhiễu và xung lỗi, được định tuyến qua một nhánh xử lý lai kết hợp cấu trúc CNN cục bộ và mạng không gian trạng thái chọn lọc Mamba nhằm khai thác tối đa ngữ cảnh thời gian.

Cơ chế Vá mảnh Thời gian dạng Tích chập (CNN Patch Embedding)

Để nén chiều dài chuỗi đầu vào và tăng cường thiên kiến quy nạp không-thời gian cục bộ, chuỗi thành phần mùa vụ được phân tách thành các mảnh thời gian (patches) kích thước  với bước nhảy dịch chuyển . Số lượng mảnh thời gian  được xác định qua công thức toán học rời rạc:

Quá trình trích xuất và chiếu các mảnh này vào không gian ẩn chiều  được thực thi song song bằng một tầng tích chập một chiều () trượt dọc theo trục thời gian của chuỗi với số kênh đầu ra bằng , kích thước nhân lọc bằng  và bước nhảy bằng :

Ý nghĩa vật lý của khối nhúng tích chập này là vận hành như một bộ lọc thông dải cục bộ thích ứng, giúp làm mịn dữ liệu và nén toàn bộ các mẫu xung biến động động học trong một thời gian ngắn thành một vector đại diện có mật độ thông tin cao.

Cơ chế Độc lập Kênh (Channel Independence — CI)

Nhằm triệt tiêu hiện tượng rò rỉ và lan truyền nhiễu xuyên kênh giữa các trục cảm biến khác nhau, chiều kích thước lô huấn luyện () và chiều kênh () được làm phẳng để chuyển đổi cấu trúc tensor không gian ẩn:

Ràng buộc này buộc bộ mã hóa phía sau phải xử lý dữ liệu từ mỗi cảm biến như một thực thể độc lập chuỗi đơn, bảo toàn nguyên vẹn các đặc tính động học đặc trưng của từng trục đo cơ học.

Khối lai Mamba-CNN (Hybrid Mamba-CNN Block)

Vectơ ẩn  sau đó được truyền qua các khối mã hóa không gian trạng thái chọn lọc lai tích chập. Tại mỗi khối, tín hiệu chiếu ẩn  tại bước thời gian mảnh  được dẫn qua một nhánh tích chập một chiều cục bộ (Local 1D CNN) với kích thước nhân lọc  và hàm kích hoạt phi tuyến SiLU để triệt tiêu nhiễu đo lường ngẫu nhiên:

Chuỗi đặc trưng đã làm sạch  đóng vai trò là toán tử điều hướng đầu vào cho mô hình không gian trạng thái tuyến tính chọn lọc (), thực hiện cập nhật trạng thái ẩn liên tục dựa trên các ma trận hệ số biến đổi phụ thuộc dữ liệu:

Việc tích hợp lớp tích chập CNN ngay trước bước quét tuyến tính của Mamba thiết lập một bộ lọc nhiễu cục bộ vững chắc, ngăn chặn hiện tượng bão hòa trạng thái ẩn của mô hình khi đối mặt với mật độ nhiễu trắng công nghiệp lớn.

Đầu dự báo mùa vụ (Seasonal Forecasting Head)

Biểu diễn context toàn cục sau khi quét qua mạng Mamba, ký hiệu là , được chuyển đến đầu dự báo chuyên dụng để chiếu ngược về không gian miền thời gian của cửa sổ dự báo tương lai:

Khối Trích xuất Đặc trưng Vật lý Thống kê Dẫn đường (Physics-Informed Statistical Head)

Để bổ khuyết cho các không gian ẩn hướng dữ liệu thuần túy vô hướng vốn vận hành như các hộp đen, một khối Stats Head vật lý được cấu trúc nhằm nhúng các bộ mô tả cơ học tường minh vào mạng. Khối này thực hiện trích xuất một vectơ đặc trưng thống kê miền thời gian 8 chiều từ cửa sổ lịch sử thô ban đầu  dựa trên hệ phương trình toán học cơ học:

Giá trị Trung bình (Mean) (Chỉ thị độ lệch tâm và dịch chuyển hằng số một chiều):

Độ lệch Chuẩn (Standard Deviation) (Đại diện cho biên độ biến động năng lượng xung quanh trị trung bình):

Giá trị Hiệu dụng (Root Mean Square — RMS) (Chỉ thị cốt lõi về tổng năng lượng phá hủy cấu trúc vật lý):

Biên độ Đỉnh-Đỉnh (Peak-to-Peak) (Phạm vi va đập và giới hạn biên độ tuyệt đối của dao động):

Độ lệch Phân phối (Skewness) (Đo lường tính bất đối xứng của mật độ phân phối dữ liệu, nhạy cảm với vết nứt sớm):

Độ nhọn Phân phối (Kurtosis) (Chỉ thị nhạy cảm bậc nhất đối với các xung va đập đột biến khi xuất hiện lỗi rỗ bề mặt):

Hệ số Đỉnh (Crest Factor) (Tỷ số giữa giá trị đỉnh tuyệt đối và giá trị hiệu dụng, biểu thị độ nhọn của xung va đập):

Hệ số Hình dạng (Shape Factor) (Tỷ số giữa giá trị RMS và giá trị trung bình tuyệt đối, phản ánh biên dạng của sóng tín hiệu):

Vectơ đặc trưng vật lý 8 chiều  này được chuẩn hóa qua tầng BatchNormalization và chuyển đổi tuyến tính trước khi thực hiện phép nối (concatenate) trực tiếp với vectơ không gian ẩn của luồng Mamba:

Cấu trúc này mang tính modulable linh hoạt, được điều khiển bởi tham số cấu hình hệ thống . Nếu cấu hình , luồng trích xuất này sẽ tự động được ngắt bỏ để trả về nhánh dự báo Mamba mùa vụ nguyên bản.

Khối Hòa trộn Hai Nhánh Tự học (Learnable Dual-Stream Mixing Module)

Để vượt qua các giới hạn của phép cộng tích hợp trực tiếp vốn cố định tỷ trọng đóng góp của các thành phần, khung phương pháp đề xuất triển khai một mô-đun hòa trộn tự học dựa trên hệ số trọng số động  được tối ưu hóa độc lập cho từng kênh cảm biến :

Trong đó  đại diện cho hàm kích hoạt Sigmoid toán học nhằm ràng buộc chặt chẽ miền giá trị của hệ số trong khoảng (0, 1), và  là tham số có khả năng tự cập nhật đạo hàm trong quá trình lan truyền ngược tương ứng với mỗi kênh cảm biến.

Cơ chế hòa trộn tự học này cho phép mạng tối ưu hóa tự động điều chỉnh linh hoạt tỷ trọng đóng góp giữa xu hướng năng lượng mài mòn dài hạn (thành phần trend) và các dao động biến động xung kích đột biến ngắn hạn (thành phần seasonal) dựa trên đặc tính vật lý riêng biệt của từng vị trí đặt cảm biến gia tốc, tối ưu hóa độ trung thực của chuỗi dự báo tổng hợp .

Tính Điểm Bất thường và Xác định Ngưỡng động POT-EVT không Rò rỉ Dữ liệu (Anomaly Scoring and Leakage-Free POT Thresholding)

Sau khi thu được chuỗi dự báo tổng hợp , điểm số bất thường (Anomaly Score) tại mỗi cửa sổ thời gian  được định nghĩa chính thức bằng sai số bình phương trung bình (Mean Squared Error — MSE) trên toàn bộ các cảm biến và bước thời gian của horizon :

Để thiết lập ranh giới quyết định động trực tuyến một cách khách quan, Lý thuyết giá trị cực biên (Extreme Value Theory — EVT) thông qua kỹ thuật vượt ngưỡng (Peak Over Threshold — POT) được tích hợp trực tiếp vào pha suy luận của mô hình. Quy trình tính toán tuân thủ nghiêm ngặt ràng buộc không rò rỉ dữ liệu (leakage-free) bằng cách chỉ thực hiện khớp hàm mật độ xác suất phân phối trên tập sai số dự báo thuộc phân đoạn dữ liệu khỏe mạnh lịch sử ban đầu, cô lập hoàn toàn khỏi mọi thông tin suy thoái tương lai. Tiến trình thực thi gồm các bước toán học rời rạc:

Xác định một ngưỡng neo ban đầu  bằng cách trích xuất phân vị cao (ví dụ: 98%) của chuỗi sai số dự báo thu được từ tập dữ liệu vận hành lành mạnh cơ sở.

Lọc và thu thập tập hợp các giá trị vượt ngưỡng cực biên dương: .

Khớp tập hợp các giá trị vượt ngưỡng  vào Phân phối Pareto Tổng quát (Generalized Pareto Distribution — GPD) nhằm mô tả toán học chính xác phân phối phần đuôi của sai số:

Trong đó  và  lần lượt là tham số hình dáng (shape parameter) và tham số quy mô (scale parameter) được tối ưu hóa qua phương pháp ước lượng hợp lý cực đại (Maximum Likelihood Estimation — MLE).

Tính toán ngưỡng quyết định động cuối cùng  ứng với một xác suất mục tiêu cảnh báo cực nhỏ  (ví dụ: ):

Trong đó  đại diện cho tổng số lượng mẫu quan sát cơ sở, và  là số lượng mẫu thực tế vượt qua ngưỡng neo  ban đầu.

Trong giai đoạn giám sát trực tuyến thời gian thực, bất kỳ cửa sổ thời gian nào có điểm số bất thường thỏa mãn điều kiện logic  sẽ lập tức được hệ thống phân loại là trạng thái bất thường cấu trúc, đảm bảo tính thực tiễn cao và triệt tiêu hoàn toàn tỷ lệ báo động giả do nhiễu động vận hành.

Experiments Design

Mô tả Tập dữ liệu và Quy trình Phân chia Thời gian

Quá trình kiểm chứng thực nghiệm cho khung phương pháp đề xuất được thực hiện thông qua tập dữ liệu vòng bi của Đại học Paderborn, một bộ dữ liệu tiêu chuẩn để đánh giá các giải thuật chẩn đoán lỗi vòng bi. Thiết lập thực nghiệm cô lập một tập con chuyên biệt các mã vòng bi để đánh giá toàn diện mô hình dưới cả kịch bản đồng nhất và biến đổi điều kiện vận hành. Mã thiết bị kiểm thử được chọn là dòng vòng bi cầu 61806-2RS, sở hữu đường kính trong 30 mm, đường kính ngoài 42 mm, độ rộng cấu trúc 7 mm, và được tích hợp 19 viên bi thép bên trong. Trong các chu kỳ thử nghiệm run-to-failure, cụm vòng bi phải chịu các điều kiện vận hành động học phức tạp, nơi tốc độ quay của trục, biên độ tải trọng tĩnh và biên độ tải trọng động siêu đặt được lấy mẫu ngẫu nhiên từ phân phối đều tĩnh nhằm mô phỏng môi trường cơ học khắc nghiệt trong thực tế. Để thu thập thông tin định hướng không gian của quá trình suy thoái cấu trúc mà không gây rò rỉ thông tin chéo, hai cảm biến gia tốc đơn hướng được gá đặt trực tiếp trên vỏ bọc cụm vòng bi, theo dõi sát hướng ngang phía sau (Kênh cảm biến A) và hướng mặt trước (Kênh cảm biến C). Tín hiệu rung động thô được số hóa ở tần số lấy mẫu siêu cao đạt 128 kHz cho các vòng đời ban đầu và 64 kHz cho các giai đoạn vận hành kéo dài phía sau. Trước khi cắt lát dữ liệu chuỗi thời gian, một bộ lọc thông cao Butterworth nhân quả được thực thi với tần số cắt cắt định hình tại 2.000 Hz nhằm lọc bỏ hoàn toàn các dao động tần số thấp từ hoạt động quay của trục và nhiễu nền công nghiệp.

Nhằm đảm bảo một giao thức huấn luyện không rò rỉ dữ liệu phản ánh chính xác các ràng buộc triển khai công nghiệp thực tế, một tập con chuyên biệt gồm chính xác 10 vòng bi tiêu biểu trong tổng số 17 vòng đời thử nghiệm đã được cô lập. Ma trận huấn luyện khai thác luồng tín hiệu tổng hợp từ các vòng bi B02, B05, B08, B10, B11, và B17. Khả năng khái quát hóa và độ chính xác chẩn đoán bất thường không giám sát được kiểm chứng độc lập trên nhóm vòng bi kiểm thử mở rộng bao gồm B01, B03, B04, B08, B10, B12, và B17. Trên quỹ đạo thời gian của từng thiết bị, một quy trình phân chia thời gian nghiêm ngặt được thực hiện: 5% thời lượng đầu tiên của chuỗi tín hiệu bị loại bỏ nhằm loại trừ các hành vi xung đột biến khởi động ban đầu. Giai đoạn huấn luyện không giám sát được giới hạn nghiêm ngặt trong pha vận hành khỏe mạnh giai đoạn đầu, khai thác cửa sổ dữ liệu từ mốc 5% đến 40% tổng vòng đời ghi nhận, nơi nhãn trạng thái được xác định bằng không (), cô lập hoàn toàn không gian ẩn khỏi các biên dạng lỗi cấu trúc. Phân đoạn còn lại từ mốc lằn ranh 40% đến khi thiết bị hư hỏng hoàn toàn được dành riêng cho tác vụ kiểm thử trực tuyến và hiệu chuẩn ngưỡng ra quyết định, thực thi sự phân tách thời gian tuyệt đối. Các chuỗi giám sát liên tục được cấu trúc qua giao thức cửa sổ trượt với chiều dài lịch sử quan sát () bằng 4.096 điểm, khoảng thời gian dự báo tương lai () bằng 1.024 điểm, dịch chuyển với bước nhảy cửa sổ cố định là 1.024 điểm.

Các Mô hình Đối chứng và Giao thức Đồng bộ Ngân sách Tham số

Để chứng minh sự vượt trội về mặt cấu trúc của kiến trúc lai Mamba-CNN, quá trình đánh giá được đối chiếu trực tiếp với năm mô hình chuỗi thời gian học sâu đại diện cho các trường phái hồi quy, tích chập, cơ chế chú ý và không gian trạng thái chọn lọc:

Long Short-Term Memory (LSTM): Mạng hồi quy tiêu chuẩn đại diện cho trường phái mô hình hóa các bước chuyển dịch thời gian tuần tự.

Temporal Convolutional Network (TCN): Mạng tích chập thời gian sử dụng các bộ lọc giãn để đánh giá hiệu năng của trường đón nhận cục bộ song song.

ModernTCN: Kiến trúc tích chập hiện đại hóa phân tách các toán tử theo chiều sâu và chiều rộng nhằm tối đa hóa năng lượng biểu diễn đặc trưng.

PatchTST: Mô hình Transformer tiên tiến vận hành dựa trên cơ chế vá mảnh thời gian và ràng buộc độc lập kênh để đạt độ phức tạp tuyến tính.

SimpleMamba: Kiến trúc không gian trạng thái chọn lọc tiêu chuẩn vận hành cô lập, không tích hợp các nhánh tích chập hoặc bộ mô tả thống kê vật lý, đóng vai trò làm cấu phần kiểm chứng ablation.

Một lỗi phổ biến trong việc thiết kế thực nghiệm là sự phân bổ không đồng đều về quy mô mô hình, dẫn đến việc mạng đề xuất vượt trội hơn đối chứng chỉ do sở hữu lượng tham số lớn hơn. Nhằm đảm bảo tính công bằng học thuật nghiêm ngặt, một giao thức đồng bộ ngân sách tham số tự động được kích hoạt (auto_scale_baselines: true). Dưới ràng buộc này, số chiều ẩn và số tầng của cả năm mô hình đối chứng sẽ tự động co giãn để cân bằng tổng lượng tham số huấn luyện tương đương với mô hình đề xuất HybridMambaCNN. Do đó, mọi sự chênh lệch về hiệu năng thu được sẽ được cô lập hoàn toàn vào tính hiệu quả của cấu trúc kiến trúc mạng thay vì sự bất cân bằng về dung lượng mô hình thô.

Hạ tầng Huấn luyện và Quy trình Tối ưu hóa

Quá trình tối ưu hóa các mạng chuỗi được thực hiện trong một khung thời gian nghiêm ngặt gồm 10 epochs với kích thước lô huấn luyện (batch size) bằng 128 mẫu trên hệ thống tăng tốc phần cứng vận hành bởi GPU kích hoạt CUDA. Tốc độ học toàn cục được thiết lập cố định ở giá trị . Thay vì sử dụng hàm sai số bình phương trung bình (MSE) truyền thống trong vòng lặp cập nhật đạo hàm, hàm mất mát toàn cục được cấu trúc dựa trên hệ hàm mất mát Huber loss. Đối với các trường hợp sai số dự báo tuyệt đối nằm trong ngưỡng giới hạn  thỏa mãn điều kiện , hình phạt toán học được định nghĩa là:

Ngược lại, đối với các trường hợp sai lệch dị thường lớn vượt ngưỡng thỏa mãn , hình phạt tuyến tính được cấu trúc như sau:

Việc lựa chọn hàm mất mát Huber được định hướng trực tiếp bởi đặc tính vật lý của dữ liệu rung động công nghiệp. Tín hiệu công nghiệp tần số cao thường xuất hiện các đỉnh xung dị thường non-Gaussian cường độ lớn do va đập cơ học ngoại biên hoặc lỗi phần cứng cảm biến. Hàm mất mát MSE tiêu chuẩn sẽ bình phương các sai số này, buộc quá trình tối ưu hóa phải điều chỉnh trọng số quá mức theo các nhiễu động ngẫu nhiên, làm mất tính ổn định của không gian ẩn khỏe mạnh. Hàm Huber tích hợp một ranh giới phạt tuyến tính cho các sai số vượt ngưỡng , đảm bảo mô hình duy trì độ ổn định cấu trúc trước các xung nhiễu đột biến trong khi vẫn bảo toàn hành vi hội tụ bình phương mượt mà cho các biến động sai số thông thường.

Chiến lược Đánh giá và Định nghĩa Điểm số Bất thường

Trạng thái sức khỏe cấu trúc của vòng bi được theo dõi liên tục thông qua sai lệch dự báo của cửa sổ kế tiếp, đóng vai trò là chỉ số đại diện cho bài toán phát hiện bất thường không giám sát. Tại mỗi bước thời gian đánh giá , điểm số bất thường  được tính toán chính thức bằng sai số bình phương trung bình (MSE) trên toàn bộ  kênh cảm biến và chiều dài khoảng dự báo  ():

Khi các cấu phần cơ học vận hành trong trạng thái ổn định ổn định, không gian ẩn của các tầng Mamba-CNN lai sẽ dự báo tín hiệu tương lai với sai lệch tối thiểu, giữ cho điểm số bất thường nằm dưới hạn định. Khi xuất hiện các hư tổn cấu trúc thô, năng lực dự báo chuỗi sẽ suy giảm đột ngột, khiến chỉ số  tăng vọt vượt trội.

Để đánh giá toàn diện hiệu năng chẩn đoán, kết quả được phân tích qua hệ thống chỉ số kiểm định PHM công nghiệp. Precision đo lường tỷ lệ cảnh báo chính xác trên tổng số cảnh báo được phát ra nhằm giảm thiểu chi phí bảo trì dư thừa. Recall đánh giá tỷ lệ bắt giữ thành công các dấu hiệu chớm lỗi cấu trúc sớm. Chỉ số F1-score thiết lập giá trị trung bình điều hòa để tổng hợp sự cân bằng động giữa hai đại lượng trên. Diện tích dưới đường cong ROC (AUROC) và diện tích dưới đường cong Precision-Recall (AUPRC) đánh giá hiệu năng độc lập với giá trị ngưỡng quyết định, trong đó AUPRC đóng vai trò là thước đo vàng tối cao do đặc tính mất cân bằng trạng thái cực đoan (class imbalance) bởi các sự cố bất thường luôn chiếm tỷ lệ rất nhỏ trong suốt vòng đời dài hạn của thiết bị. Cuối cùng, tỷ lệ báo động giả (FAR) đo lường tần suất phát ra các cảnh báo sai, đóng vai trò là chỉ số tối ưu hóa chi phí vận hành cốt lõi.

* Lưu ý: Do các giới hạn về tài nguyên phần cứng tính toán cục bộ và giao thức truy cập kho lưu trữ, một tập con thu gọn và mang tính đại diện diện rộng của tập dữ liệu vòng bi Đại học Paderborn được lựa chọn sử dụng trong các lượt thực nghiệm. Ranh giới cấu trúc này được thiết lập một cách chủ ý nhằm xây dựng một đường cơ sở so sánh cục bộ có mật độ thông tin cao, đảm bảo tính lặp lại thực nghiệm tuyệt đối dưới một ngân sách tham số giới hạn.

Results

Phân tích Phân tách Chuỗi Thời gian và Đặc tính Đa Quy mô (Waveform Decomposition Dynamics)

Figure 1: Waveform Decomposition

Quá trình phân tách tín hiệu rung động rời rạc rời rạc thông qua mô-đun phân tách chuỗi thời gian thích ứng được đánh giá trực quan trên ba giai đoạn tiêu biểu thuộc vòng đời của vòng bi B03: trạng thái khỏe mạnh ban đầu (Healthy — M0001), giai đoạn giữa vòng đời (Mid-life — M0308), và thời điểm chớm phát sinh lỗi cấu trúc (Fault Onset — M0558). Như được minh chứng tại Fig. 1, việc cấu hình nhân lọc trung bình trượt kích thước mở rộng  đảm bảo thành phần xu hướng () duy trì một biên dạng phẳng, mượt mà và triệt tiêu hoàn toàn các dao động cục bộ tần số cao. Sự ổn định này khẳng định rằng nhánh xu hướng chỉ tập trung bắt giữ mức dịch chuyển hằng số một chiều và động học suy thoái lũy tiến chậm của hệ thống cơ khí.

Ngược lại, thành phần mùa vụ () biểu diễn sự trùng khớp gần như tuyệt đối về mặt hình thái học đối với tín hiệu thô ban đầu () xuyên suốt hai giai đoạn đầu của chu kỳ sống. Tại thời điểm chớm lỗi (M0558), một sự bùng nổ biên độ mạnh mẽ được ghi nhận trên luồng seasonal, nơi giá trị hiệu dụng (Root Mean Square — RMS) tăng vọt từ 0.059 V lên 0.088 V, trong khi năng lượng của thành phần xu hướng vẫn được giữ ở mức danh định cực thấp (RMS đạt 0.006 V). Hiện tượng đi xuống cục bộ mang tính âm của nhánh trend tại thời điểm này được xác định là một hiệu ứng phụ kỹ thuật do sự xuất hiện của các xung đột biến năng lượng cao, hoàn toàn không ảnh hưởng đến tính ổn định toán học toàn cục. Kết quả này xác nhận giải thuật phân tách chuỗi đã cô lập thành công các tín hiệu xung va đập hư tổn ra khỏi đường nền công nghiệp nặng.

Minh chứng Miền Tần số và Sự Phân tách Năng lượng Phổ (Frequency-Domain Verification)

Figure 2: PSD Frequency Evidence

Để kiểm định tính đúng dải tần của mô-đun tiền xử lý và phân tách, mật độ phổ công suất (Power Spectral Density — PSD) của các luồng đặc trưng được phân tích đối chứng. Tại Fig. 2, một sự phân tách phổ sâu từ 6 đến 8 bậc biên độ được ghi nhận một cách rõ ràng trong dải tần số cao từ 1000 Hz đến 6000 Hz giữa cấu phần trend và cấu phần seasonal. Chỉ số đo lường tỷ lệ năng lượng phổ chứng minh thành phần mùa vụ chiếm ưu thế tuyệt đối với giá trị gấp 28 lần năng lượng của thành phần xu hướng, đánh dấu bước cải thiện vượt trội so với các cấu hình nhân lọc kích thước hẹp truyền thống.

Sự hiện diện của các vùng chồng lấn phổ tại dải tần số thấp dưới 200 Hz được xác định là do các dao động tần số thấp không thể tránh khỏi từ trục quay của động cơ motor nền. Tuy nhiên, tính chất cơ học của lỗi được xác thực tường minh tại biểu đồ biến đổi Fourier nhanh (Fast Fourier Transform — FFT) của thành phần mùa vụ trong trạng thái lỗi. Một đỉnh xung năng lượng vượt trội xuất hiện cô lập và chuẩn xác tại tần số 108 Hz, trùng khớp hoàn toàn với tần số hư hại đường rãnh vòng trong BPFI của thiết bị cơ khí. Bằng chứng miền tần số này khẳng định nhánh seasonal xử lý bởi bộ mã hóa Mamba-CNN đang phản ánh trực tiếp các vi xung động động học của khuyết tật cơ học thô thay vì học các nhiễu trắng ngẫu nhiên từ môi trường công nghiệp nặng.

Quỹ đạo Suy thoái Dài hạn và Phân tích Không gian Ẩn (Longitudinal Evolution and Dimensionality Analysis)

Figure 3: Longitudinal RMS

Figure 4: PCA Dimensionality

Figure 5: Architecture Justification

Quá trình tiến triển lỗi trên toàn bộ vòng đời gồm 614 tệp dữ liệu run-to-failure của vòng bi B03 được theo dõi liên tục thông qua các chỉ số năng lượng miền thời gian. Như được hiển thị tại Fig. 3, giá trị RMS của thành phần mùa vụ duy trì tính ổn định tuyệt đối trong suốt 89% giai đoạn đầu của vòng đời, và lập tức ghi nhận một cú nhảy vọt (spike) chuẩn xác tại thời điểm M0547, liên tục dao động trong dải biên độ cao từ 0.15 V đến 0.35 V phản ánh sự phát triển nghiêm trọng của vết nứt thớ thép. Quỹ đạo biến thiên của tín hiệu thô gốc thể hiện sự đồng thuận đồng bộ đối với nhánh seasonal, trong khi RMS của nhánh xu hướng tiệm cận sát mức 0 suốt phần lớn chu kỳ vận hành. Kết quả này hợp thức hóa việc áp dụng một nhánh dự báo tuyến tính đơn giản cho thành phần xu hướng nhằm tối ưu hóa ngân sách tài nguyên tính toán.

Sự cần thiết của cấu trúc luồng kép song song lai Mamba-CNN được chứng minh bằng định lượng toán học thông qua phân tích suy giảm chiều phương sai cumulative tại Fig. 4. Thành phần xu hướng đạt tới 90% tổng phương sai giải thích chỉ với 2 thành phần chính (Principal Components — PCs), minh chứng cho một cấu trúc đa tạp phẳng có số chiều cực thấp (nearly planar manifold). Ngược lại, thành phần mùa vụ thể hiện các đặc tính động học phi tuyến tính đa chiều cực kỳ phức tạp, nơi việc sử dụng đến 30 thành phần chính mới chỉ giải thích được xấp xỉ 83% lượng phương sai tổng thể. Tỷ lệ mở rộng không gian chiều giữa hai cấu phần đạt mức . As empirically demonstrated on the B03 run-to-failure dataset, the trend component requires only 2 principal components to explain 90% of its variance, while the seasonal component requires more than 30 components to reach 83% (dimensionality ratio: ). This disparity directly motivates the proposed dual-stream architecture.

Cuối cùng, Fig. 5 thiết lập một ranh giới biện chứng hoàn chỉnh về mặt kiến trúc hệ thống. Chỉ số độ nhọn (Kurtosis) của nhánh seasonal ổn định nghiêm ngặt xung quanh giá trị 3.0 (phân phối chuẩn Gaussian) trong suốt kỷ nguyên vận hành khỏe mạnh, và lập tức bùng nổ vượt ngưỡng giá trị 4.0 tại thời điểm M0547. Đặc tính phi vi phân, non-Gaussian và xung kích mạnh này loại bỏ hoàn toàn khả năng xử lý của các mạng tuyến tính thông thường, đặt ra yêu cầu bắt buộc về một cơ chế quét tuyến tính chọn lọc phụ thuộc đầu vào của bộ mã hóa Mamba. Biểu đồ phân tán (scatter plot) giữa hai thành phần năng lượng thiết lập hai phân vùng rõ rệt: trạng thái lành mạnh hội tụ dồn cụm tại vùng gốc tọa độ (Trend , Seasonal ), trong khi trạng thái lỗi phân tán kéo dài dọc theo trục seasonal, chứng minh tính độc lập cấu trúc trong pha hiệu chuẩn ban đầu và sự đồng tiến triển hình thái (co-evolve) tại giai đoạn suy thoái cấu trúc nặng.

Hệ Thống Bảng Biểu Định Lượng (Quantitative Evidence Tables)

TABLE II QUANTITATIVE EVOLUTION AND SPECTRAL SEPARATION METRICS ACROSS FILTER KERNELS (BEARING B03)

| Đơn vị Đo lường Thực nghiệm (Experimental Metric) | Cấu hình Cũ (Kernel = 257) | Cấu hình Đề xuất (Kernel = 3457) | Ý nghĩa Vật lý / Cơ học (Physical Signification) |
| --- | --- | --- | --- |
| PCA Trend Variance Bound (90% Threshold) | 13 PCs | 2 PCs | Đa tạp phẳng, số chiều cực thấp  Hợp thức hóa nhánh Tuyến tính. |
| PCA Seasonal Variance Bound (90% Threshold) | 31 PCs | > 30 PCs (83% at 30) | Động học phi tuyến tính, đa chiều phức tạp  Yêu cầu Mamba Encoder. |
| Dimensionality Expansion Ratio (Seasonal/Trend) |  |  | Minh chứng định lượng cho Nguyên lý Tối giản Kiến trúc (Parsimony). |
| Spectral Energy Separation (Seasonal/Trend) |  |  | Triệt tiêu hoàn toàn hiện tượng nhiễu xu hướng che lấp xung va đập lỗi (Fault Masking). |
| Identified Analytical Fault Peak Frequency | None | 108 Hz | Trùng khớp hoàn toàn với tần số khuyết tật cơ học vòng trong (BPFI). |

TABLE III LIMITATION DIAGNOSTICS AUDIT AND REPRODUCIBILITY DEFENSE MATRIX

| Hiện tượng Chênh lệch (Observed Artifact) | Bản chất Kỹ thuật (Technical Nature) | Tác động Hệ thống (Systemic Impact) | Luận điểm Phòng thủ Khoa học (Scholarly Defense Argument) |
| --- | --- | --- | --- |
| Trend Negative Drift (Fig. 1 — Fault Onset Column) | Hiện tượng méo dạng dòng một chiều (DC drift artifact) khi bộ lọc trung bình trượt gặp chuỗi xung va đập biên độ cực lớn. | Hoàn toàn không gây lỗi cấu trúc mạng. | Đây là đặc tính phản ứng tự nhiên của toán tử tích phân trượt; sự xuất hiện của xu hướng âm chứng minh luồng seasonal đã rút cạn toàn bộ năng lượng xung đột biến tần số cao. |
| Low-Frequency Spectral Overlap (Fig. 2 — Below 200 Hz Region) | Sự chồng lấn dải tần do các dao động cơ học từ trục quay chính (shaft rotation) và tải trọng motor nền. | Nhiễu dải nền biên độ thấp. | Hiện tượng trùng lấn ở tần số thấp là bất khả kháng đối với mọi bộ lọc trung bình trượt (MA filter). Lỗi này được triệt tiêu hoàn toàn thông qua cơ chế Độc lập Kênh (CI) và khối tích chập cục bộ Local 1D CNN trước khi quét Mamba. |
| Visual Bounding Absence (Fig. 5 — Scatter Cluster Distribution) | Thiếu vòng elip phân định ranh giới (ellipse validation annotations) giữa cụm lành mạnh và cụm lỗi. | Giảm nhẹ tính trực quan đồ họa. | Sự phân tách phân cụm hình thái học (Healthy cụm đặc, Fault phân tán tuyến tính dọc trục Seasonal) đã quá tường minh về mặt số liệu toán học; thuật toán POT-EVT sẽ tự động hóa việc gán biên quyết định này một cách không giám sát. |

Ease of Use

A. Selecting a Template (Heading 2)

First, confirm that you have the correct template for your paper size. This template has been tailored for output on the A4 paper size. If you are using US letter-sized paper, please close this file and download the file “MSW_USltr_format”.

B. Maintaining the Integrity of the Specifications

The template is used to format your paper and style the text. All margins, column widths, line spaces, and text fonts are prescribed; please do not alter them. You may note peculiarities. For example, the head margin in this template measures proportionately more than is customary. This measurement and others are deliberate, using specifications that anticipate your paper as one part of the entire proceedings, and not as an independent document. Please do not revise any of the current designations.

Prepare Your Paper Before Styling

Before you begin to format your paper, first write and save the content as a separate text file. Keep your text and graphic files separate until after the text has been formatted and styled. Do not use hard tabs, and limit use of hard returns to only one return at the end of a paragraph. Do not add any kind of pagination anywhere in the paper. Do not number text heads-the template will do that for you.

Finally, complete content and organizational editing before formatting. Please take note of the following items when proofreading spelling and grammar:

Abbreviations and Acronyms

Define abbreviations and acronyms the first time they are used in the text, even after they have been defined in the abstract. Abbreviations such as IEEE, SI, MKS, CGS, sc, dc, and rms do not have to be defined. Do not use abbreviations in the title or heads unless they are unavoidable.

Units

Use either SI (MKS) or CGS as primary units. (SI units are encouraged.) English units may be used as secondary units (in parentheses). An exception would be the use of English units as identifiers in trade, such as “3.5-inch disk drive.”

Avoid combining SI and CGS units, such as current in amperes and magnetic field in oersteds. This often leads to confusion because equations do not balance dimensionally. If you must use mixed units, clearly state the units for each quantity that you use in an equation.

Do not mix complete spellings and abbreviations of units: “Wb/m2” or “webers per square meter,” not “webers/m2.” Spell units when they appear in text: “...a few henries,” not “...a few H.”

Identify applicable sponsor/s here. If no sponsors, delete this text box (sponsors).

Use a zero before decimal points: “0.25,” not “.25.” Use “cm3,” not “cc.” (bullet list)

Equations

The equations are an exception to the prescribed specifications of this template. You will need to determine whether or not your equation should be typed using either the Times New Roman or the Symbol font (please no other font). To create multileveled equations, it may be necessary to treat the equation as a graphic and insert it into the text after your paper is styled.

Number equations consecutively. Equation numbers, within parentheses, are to position flush right, as in (1), using a right tab stop. To make your equations more compact, you may use the solidus ( / ), the exp function, or appropriate exponents. Italicize Roman symbols for quantities and variables, but not Greek symbols. Use a long dash rather than a hyphen for a minus sign. Punctuate equations with commas or periods when they are part of a sentence, as in

A+b= y                                                                                                                              (1)

Note that the equation is centered using a center tab stop. Be sure that the symbols in your equation have been defined before or immediately following the equation. Use “(1),” not “Eq. (1)” or “equation (1),” except at the beginning of a sentence: “Equation (1) is ...”

Some Common Mistakes

The word “data” is plural, not singular.

The subscript for the permeability of vacuum 0, and other common scientific constants, is zero with subscript formatting, not a lowercase letter “o.”

In American English, commas, semi-/colons, periods, question and exclamation marks are located within quotation marks only when a complete thought or name is cited, such as a title or full quotation. When quotation marks are used, instead of a bold or italic typeface, to highlight a word or phrase, punctuation should appear outside of the quotation marks. A parenthetical phrase or statement at the end of a sentence is punctuated outside of the closing parenthesis (like this). (A parenthetical sentence is punctuated within the parentheses.)

A graph within a graph is an “inset,” not an “insert.” The word alternatively is preferred to the word “alternately” (unless you really mean something that alternates).

Do not use the word “essentially” to mean “approximately” or “effectively.”

In your paper title, if the words “that uses” can accurately replace the word using, capitalize the “u”; if not, keep using lower-cased.

Be aware of the different meanings of the homophones “affect” and “effect,” “complement” and “compliment,” “discreet” and “discrete,” “principal” and “principle.”

Do not confuse “imply” and “infer.”

The prefix “non” is not a word; it should be joined to the word it modifies, usually without a hyphen.

There is no period after the “et” in the Latin abbreviation “et al.”

The abbreviation “i.e.” means “that is,” and the abbreviation “e.g.” means “for example.”

An excellent style manual for science writers is [7].

Using the Template

After the text edit has been completed, the paper is ready for the template. Duplicate the template file by using the Save As command, and use the naming convention prescribed by your conference for the name of your paper. In this newly created file, highlight all of the contents and import your prepared text file. You are now ready to style your paper; use the scroll down window on the left of the MS Word Formatting toolbar.

Authors and Affiliations

The template is designed so that author affiliations are not repeated each time for multiple authors of the same affiliation. Please keep your affiliations as succinct as possible (for example, do not differentiate among departments of the same organization). This template was designed for two affiliations.

For author/s of only one affiliation (Heading 3): To change the default, adjust the template as follows.

Selection (Heading 4): Highlight all author and affiliation lines.

Change number of columns: Select the Columns icon from the MS Word Standard toolbar and then select “1 Column” from the selection palette.

Deletion: Delete the author and affiliation lines for the second affiliation.

For author/s of more than two affiliations: To change the default, adjust the template as follows.

Selection: Highlight all author and affiliation lines.

Change number of columns: Select the “Columns” icon from the MS Word Standard toolbar and then select “1 Column” from the selection palette.

Highlight author and affiliation lines of affiliation 1 and copy this selection.

Formatting: Insert one hard return immediately after the last character of the last affiliation line. Then paste down the copy of affiliation 1. Repeat as necessary for each additional affiliation.

Reassign number of columns: Place your cursor to the right of the last character of the last affiliation line of an even numbered affiliation (e.g., if there are five affiliations, place your cursor at end of fourth affiliation). Drag the cursor up to highlight all of the above author and affiliation lines. Go to Column icon and select “2 Columns”. If you have an odd number of affiliations, the final affiliation will be centered on the page; all previous will be in two columns.

Identify the Headings

Headings, or heads, are organizational devices that guide the reader through your paper. There are two types: component heads and text heads.

Component heads identify the different components of your paper and are not topically subordinate to each other. Examples include ACKNOWLEDGMENTS and REFERENCES, and for these, the correct style to use is “Heading 5.” Use “figure caption” for your Figure captions, and “table head” for your table title. Run-in heads, such as “Abstract,” will require you to apply a style (in this case, italic) in addition to the style provided by the drop down menu to differentiate the head from the text.

Text heads organize the topics on a relational, hierarchical basis. For example, the paper title is the primary text head because all subsequent material relates and elaborates on this one topic. If there are two or more sub-topics, the next level head (uppercase Roman numerals) should be used and, conversely, if there are not at least two sub-topics, then no subheads should be introduced. Styles named “Heading 1,” “Heading 2,” “Heading 3,” and “Heading 4” are prescribed.

Figures and Tables

Positioning Figures and Tables: Place figures and tables at the top and bottom of columns. Avoid placing them in the middle of columns. Large figures and tables may span across both columns. Figure captions should be below the figures; table heads should appear above the tables. Insert figures and tables after they are cited in the text. Use the abbreviation “Fig. 1,” even at the beginning of a sentence.

Table 1. Table Styles

| Table Head | Table Column Head |  |  |
| --- | --- | --- | --- |
|  | Table column subhead | Subhead | Subhead |
| copy | More table copya |  |  |

*Sample of a Table footnote. (Table footnote)

We suggest that you use a text box to insert a graphic (which is ideally a 300 dpi resolution TIFF or EPS file with all fonts embedded) because this method is somewhat more stable than directly inserting a picture.To have non-visible rules on your frame, use the MSWord “Format” pull-down menu, select Text Box > Colors and Lines to choose No Fill and No Line.We suggest that you use a text box to insert a graphic (which is ideally a 300 dpi resolution TIFF or EPS file with all fonts embedded) because this method is somewhat more stable than directly inserting a picture.To have non-visible rules on your frame, use the MSWord “Format” pull-down menu, select Text Box > Colors and Lines to choose No Fill and No Line.

Figure 1. Example of a figure caption (figure caption)

Figure Labels: Use 8 point Times New Roman for Figure labels. Use words rather than symbols or abbreviations when writing Figure axis labels to avoid confusing the reader. As an example, write the quantity “Magnetization,” or “Magnetization, M,” not just “M.” If including units in the label, present them within parentheses. Do not label axes only with units. In the example, write “Magnetization (A/m)” or “Magnetization (A ( m(1),” not just “A/m.” Do not label axes with a ratio of quantities and units. For example, write “Temperature (K),” not “Temperature/K.”

Acknowledgment

The preferred spelling of the word “acknowledgment” in America is without an “e” after the “g.” Avoid the stilted expression “one of us (R. B. G.) thanks ...”.  Instead, try “R. B. G. thanks...”. Put sponsor acknowledgments in the unnumbered footnote on the first page.

References

The template will number citations consecutively within brackets [1]. The sentence punctuation follows the bracket [2]. Refer simply to the reference number, as in [3]—do not use “Ref. [3]” or “reference [3]” except at the beginning of a sentence: “Reference [3] was the first ...”

Number footnotes separately in superscripts. Place the actual footnote at the bottom of the column in which it was cited. Do not put footnotes in the reference list. Use letters for table footnotes.

Unless there are six authors or more give all authors’ names; do not use “et al.”. Papers that have not been published, even if they have been submitted for publication, should be cited as “unpublished” [4]. Papers that have been accepted for publication should be cited as “in press” [5]. Capitalize only the first word in a paper title, except for proper nouns and element symbols.

For papers published in translation journals, please give the English citation first, followed by the original foreign-language citation [6].

G. Eason, B. Noble, and I.N. Sneddon, “On certain integrals of Lipschitz-Hankel type involving products of Bessel functions,” Phil. Trans. Roy. Soc. London, vol. A247, pp. 529-551, April 1955. (references)

J. Clerk Maxwell, A Treatise on Electricity and Magnetism, 3rd ed., vol. 2. Oxford: Clarendon, 1892, pp.68-73.

I.S. Jacobs and C.P. Bean, “Fine particles, thin films and exchange anisotropy,” in Magnetism, vol. III, G.T. Rado and H. Suhl, Eds. New York: Academic, 1963, pp. 271-350.

K. Elissa, “Title of paper if known,” unpublished.

R. Nicole, “Title of paper with only first word capitalized,” J. Name Stand. Abbrev., in press.

Y. Yorozu, M. Hirano, K. Oka, and Y. Tagawa, “Electron spectroscopy studies on magneto-optical media and plastic substrate interface,” IEEE Transl. J. Magn. Japan, vol. 2, pp. 740-741, August 1987 [Digests 9th Annual Conf. Magnetics Japan, p. 301, 1982].

M. Young, The Technical Writer’s Handbook. Mill Valley, CA: University Science, 1989.

Author 1 short CV and phorograph

Author 2 short CV and phorograph
