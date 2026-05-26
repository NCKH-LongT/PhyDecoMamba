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

Mô hình hóa chuỗi học sâu trong Quản lý Sức khỏe và Dự đoán (PHM)

Các phân tích thống kê kỹ thuật thực tế khẳng định rằng tỷ lệ sự cố liên quan đến vòng bi chiếm tới 40%–50% tổng số hư hại cấu trúc của các hệ thống máy móc quay, trực tiếp dẫn đến hiện tượng dừng máy ngoài kế hoạch và gây ra những tổn thất kinh tế nghiêm trọng cho doanh nghiệp [15]. Việc xây dựng các hệ thống giám sát chẩn đoán tự động dựa trên tín hiệu rung động trong môi trường công nghiệp nặng luôn phải đối mặt với các thách thức lớn do đặc tính phân phối không dừng, động học phi tuyến tính phức tạp và mật độ nhiễu nền cao [14]. Nhằm trích xuất các đặc trưng thưa đặc trưng trong miền thời-tần một cách không giám sát, phương pháp tối ưu hóa bộ lọc Wavelet học được (learnable Wavelet) như kiến trúc DeSpaWN [1] đã chứng minh hiệu quả xử lý tín hiệu nền tảng vượt trội.

Tuy nhiên, khi chuyển dịch sang các bài toán mô hình hóa chuỗi thời gian dài, các kiến trúc học sâu truyền thống bắt đầu bộc lộ các giới hạn cố định về mặt cấu trúc thuật toán. Mạng bộ nhớ ngắn dài hạn (LSTM) phụ thuộc vào cơ chế cập nhật trạng thái ẩn tuần tự, gây ra nút thắt cổ chai tính toán nghiêm trọng, không thể thực hiện song song hóa và dễ rơi vào hiện tượng tiêu biến đạo hàm khi chiều dài chuỗi tăng cao. Mạng tích chập thời gian (TCN) giải quyết khả năng tính toán song song thông qua các bộ lọc giãn (dilated convolutions), song trường đón nhận (receptive field) của hệ thống vẫn bị giới hạn cố định bởi kích thước bộ lọc, khiến mô hình dễ bỏ sót các quỹ đạo suy thoái chu kỳ dài khi tần số lấy mẫu của cảm biến quá cao.

Mặc dù các cấu trúc hiện đại hóa mạng tích chập như ModernTCN [2] đã tối ưu hóa đáng kể hiệu suất biểu diễn thông qua việc phân tách tích chập theo chiều sâu và chiều rộng, các nghiên cứu phản biện chuyên sâu như ModernTCN Revisited [3] đã vạch trần lỗ hổng của thủ thuật "Drop Last Trick" và hiện tượng rò rỉ dữ liệu (data leakage) trong thực nghiệm gốc. Điều này đặt ra yêu cầu khắt khe về việc thiết lập các giao thức thực nghiệm trung thực và cơ chế đồng bộ ngân sách tham số công bằng. Ngoài ra, các mạng tự mã hóa (Autoencoders) áp dụng cho tác vụ nhận diện bất thường không giám sát thường gánh chịu lỗi lệch mục tiêu học (mismatch) giữa pha tối ưu hóa và pha suy luận, điều đã được khắc phục một phần bằng cơ chế học sự khác biệt ngữ nghĩa liên hợp trong mô hình TimeRCD [4]. Nghiên cứu này kế thừa tư duy của [3] và [4] bằng cách triển khai một khung dự báo chuỗi thay vì tái cấu trúc thô, đồng thời thực thi nghiêm ngặt giao thức đồng bộ hóa quy mô tham số đối chứng.

Mô hình chuỗi thời gian dựa trên cơ chế Chú ý (Transformer)

Sự phát triển của cơ chế tự chú ý (Self-Attention) trong các cấu trúc Transformer đã tạo ra những bước tiến lớn trong việc bắt giữ các mối tương quan dài hạn của chuỗi thời gian. Trong bài toán phát hiện bất thường không giám sát, kiến trúc Anomaly Transformer [5] đã đặt nền móng cho việc đo lường sự sai khác liên hợp (Association Discrepancy) giữa phân phối chú ý cục bộ và toàn cục để cô lập các điểm dị thường. Nhằm giải quyết triệt để tính chất không dừng của tín hiệu công nghiệp, mô hình TimesNet [6] đã đề xuất giải thuật biến đổi chuỗi thời gian một chiều (1D) thành cấu trúc hai chiều (2D) thông qua phân tích đa chu kỳ Fourier (FFT), cho phép khai thác đồng thời các biến động nội chu kỳ và liên chu kỳ. Tiếp đó, mô hình PatchTST [7] đã tối ưu hóa hiệu suất của Transformer bằng cách giới thiệu cơ chế vá mảnh thời gian (patching) kết hợp với ràng buộc độc lập kênh (Channel Independence - CI), thành công giảm độ phức tạp tính toán của mạng chú ý xuống bậc tuyến tính và hạn chế nhiễu xuyên kênh.

Bất chấp những đóng góp mang tính nền tảng của nhóm phương pháp này, các mô hình dựa trên Transformer vẫn tồn tại các điểm nghẽn cốt lõi khi ứng dụng vào miền PHM thực tế. Thứ nhất, độ phức tạp tính toán và bộ nhớ bậc hai () của ma trận tự chú ý nguyên bản vẫn gây áp lực lớn lên tài nguyên phần cứng khi xử lý các cửa sổ quan sát lịch sử mở rộng. Thứ hai, và quan trọng nhất, các bộ mã hóa Transformer thiếu các thiên kiến quy nạp tịnh tiến cục bộ (localized translation invariance). Khi tín hiệu vibration bị bủa vây bởi nhiễu trắng công nghiệp hoặc xung đột biến từ các thiết bị lân cận, Transformer dễ bị đánh lừa và gán nhãn sai các đỉnh nhiễu ngẫu nhiên thành các dấu hiệu suy thoái cấu trúc, dẫn đến tỷ lệ báo động giả tăng cao. Kiến trúc lai Mamba-CNN đề xuất trong nghiên cứu này khắc phục triệt để nhược điểm trên bằng cách dung hợp bộ quét chọn lọc tuyến tính với nhánh tích chập cục bộ để duy trì tính ổn định trước nhiễu.

Mô hình Không gian Trạng thái chọn lọc (Mamba) trong miền PHM

Để phá vỡ giới hạn tính toán của Transformer, khung lý thuyết gốc về mô hình không gian trạng thái chọn lọc (Mamba) do Gu và Dao đề xuất [16] đã thiết lập một cột mốc mới, đạt độ phức tạp thời gian tuyến tính () nhờ cơ chế quét chọn lọc (selective scan) phụ thuộc vào dữ liệu đầu vào. Trong miền PHM và chẩn đoán lỗi vòng bi, các biến thể ứng dụng Mamba bắt đầu xuất hiện mạnh mẽ trong giai đoạn 2025–2026. Kiến trúc FEMamba [8] tích hợp khối thích ứng MFES và mô-đun DSGR ràng buộc không gian suy thoái phục vụ cho bài toán hồi quy dự đoán tuổi thọ còn lại (RUL). Khung mô hình TFG-Mamba [9] thực hiện dung hợp miền thời gian và miền tần số thông qua cơ chế cổng Mamba điều hướng để bắt giữ xu hướng suy thoái dài hạn của máy móc quay.

Đối với các cấu trúc đa biến phức tạp, mô hình TimeMachine [10] triển khai kiến trúc Quadruple-Mamba nhằm tích hợp đồng thời cơ chế trộn kênh và độc lập kênh, trong khi kiến trúc TSC-Mamba [11] áp dụng quy trình "Phân tách - Lan truyền - Tương quan chéo" qua mô-đun CIFM bậc thấp để tối ưu hóa biểu diễn chuỗi. Nhằm hướng tới việc triển khai thực tế, mô hình LDGM (Mamba-MP) [12] hỗ trợ phân tách đặc trưng động học toàn cục và vi xung động cục bộ cho các hệ thống biên thông minh (edge deployment). Đặc biệt, nghiên cứu PG-TMT [13] (Physics-Guided Tiny-Mamba Transformer) trên tạp chí IEEE Transactions on Reliability đã chứng minh tính khả thi của việc kết hợp cấu trúc học sâu gọn nhẹ với tri thức vật lý dải tần chẩn đoán lỗi để giám sát trực tuyến thời gian thực.

Tuy nhiên, một giới hạn trung thực tồn tại xuyên suốt trong các nghiên cứu SOTA dựa trên Mamba này là chúng hầu hết được cấu trúc như các mô hình "hộp đen" hướng dữ liệu thuần túy (purely data-driven), vận hành dưới dạng phân loại có giám sát (supervised) dựa trên nhãn lỗi có sẵn hoặc hồi quy RUL. Trong kịch bản công nghiệp thực tế, dữ liệu nhãn lỗi cực kỳ khan hiếm và đắt đỏ. Các mô hình này hoàn toàn thiếu vắng một cơ chế dự báo chuỗi không giám sát được ràng buộc và dẫn đường trực tiếp bởi một khối thống kê vật lý tường minh. Nghiên cứu hiện tại giải quyết khoảng trống này bằng cách nhúng trực tiếp khối Stats Head 8 chiều để hướng dẫn không gian ẩn của Mamba, thiết lập một bài toán dự báo không giám sát thực tế.

Phương pháp thiết lập ngưỡng động và hiệu chuẩn phát hiện bất thường

Một hệ thống phát hiện bất thường hoàn chỉnh không chỉ phụ thuộc vào kiến trúc mạng mà còn bị quyết định bởi cơ chế thiết lập ngưỡng quyết định. Các ngưỡng tĩnh cố định truyền thống (như giới hạn 3-Sigma tiêu chuẩn hoặc khoảng tứ phân vị Robust/IQR) hoàn toàn thất bại dưới điều kiện vận hành thay đổi (thay đổi tốc độ, tải trọng) do phân phối dữ liệu miền đích liên tục dịch chuyển. Các mô hình thống kê động như mô hình hỗn hợp Gauss (GMM) đã được áp dụng để mô hình hóa biên quyết định, song thường thiếu cơ sở lý thuyết toán học vững chắc khi đối mặt với các giá trị cực biên nằm ngoài phân phối huấn luyện.

Việc tích hợp Lý thuyết giá trị cực biên (Extreme Value Theory - EVT) thông qua kỹ thuật vượt ngưỡng (Peak Over Threshold - POT) trực tiếp vào không gian ẩn của mô hình học sâu như nghiên cứu trong [17] đã chứng minh năng lực thiết lập ranh giới quyết định tập mở (open set) chính xác, loại bỏ các hành vi dị thường một cách hệ thống. Mặc dù vậy, các khung hiệu chuẩn hiện hành trong các tài liệu PHM thường vô tình gây ra hiện tượng rò rỉ dữ liệu (data leakage) nghiêm trọng. Lỗi này phát sinh do quá trình tối ưu hóa siêu tham số của ngưỡng được thực hiện trên toàn bộ phân phối dữ liệu validation (bao gồm cả các phân đoạn đã chớm lỗi hoặc lỗi hoàn toàn trong tương lai), khiến mô hình không thể triển khai trực tuyến trực tiếp trên các thiết bị thực tế—nơi hệ thống chỉ được phép tiếp cận luồng dữ liệu khỏe mạnh lịch sử ban đầu. Mô hình đề xuất loại bỏ hoàn toàn lỗi này bằng một quy trình hiệu chuẩn cục bộ (localized calibration) nghiêm ngặt, chỉ tính toán cấu hình POT dựa trên các phân đoạn dữ liệu có nhãn khỏe mạnh ban đầu (condition variable == 0).

Bảng Ma trận So sánh Tài liệu (Literature Matrix Table)

TABLE I COMPREHENSIVE LITERATURE MATRIX AND STRUCTURAL COMPARISON WITH THE PROPOSED PARADIGM

| Nghiên cứu (Study) | Tác vụ (Task) | Tập dữ liệu (Dataset) | Phương pháp (Method) | Điểm mạnh (Strength) | Hạn chế (Limitation) | Mối liên hệ với nghiên cứu này (Relation to this work) |
| --- | --- | --- | --- | --- | --- | --- |
| Anomaly Transformer (2022) [5] | Phát hiện bất thường không giám sát | Chuỗi thời gian đa miền tổng quát (SMD, MSD, v.v.) | Cơ chế tính toán Sự khác biệt liên hợp (Association Discrepancy) dựa trên Transformer. | Cô lập hiệu quả các dị thường chu kỳ nhờ phân phối chú ý cục bộ và toàn cục. | Độ phức tạp tính toán bậc hai (); cực kỳ nhạy cảm với nhiễu nền công nghiệp. | Mô hình đề xuất thay thế khối tự chú ý bằng bộ mã hóa Mamba để đạt độ phức tạp tuyến tính . |
| TimesNet (2023) [6] | Dự báo và phát hiện bất thường | Chuỗi thời gian không dừng (Weather, ETT, v.v.) | Biến đổi chuỗi 1D thành cấu trúc 2D thông qua phân tích đa chu kỳ Fourier (FFT) và Inception blocks. | Bắt giữ tốt các đặc tính không dừng nhờ phân tích đa chu kỳ đồng thời. | Chi phí tính toán lớn do xử lý ảnh 2D; cấu trúc CNN thuần túy thiếu khả năng quét chọn lọc chuỗi siêu dài. | Mô hình đề xuất sử dụng cơ chế phân tách chuỗi (Series Decomposition) trực tiếp trên miền thời gian để giảm chi phí FFT. |
| PatchTST (2023) [7] | Dự báo chuỗi thời gian dài | Tập dữ liệu chuỗi thời gian chuẩn (Electricity, Traffic) | Kỹ thuật vá mảnh thời gian (Patching) kết hợp ràng buộc độc lập kênh (Channel Independence). | Giảm độ phức tạp của Transformer xuống bậc tuyến tính; loại bỏ nhiễu xuyên kênh. | Hoàn toàn bỏ sót việc dung hợp tương quan giữa các kênh, làm giảm độ chính xác trong hệ thống đa cảm biến. | Mô hình đề xuất giữ cơ chế độc lập kênh ở nhánh Mamba nhưng dung hợp xuyên kênh ở đầu phát hiện (Fusion Head). |
| TFG-Mamba (2026) [9] | Dự báo xu hướng suy thoái thiết bị | Tập dữ liệu suy thoái vòng bi thực tế | Dung hợp miền thời-tần thông qua cơ chế cổng Mamba điều hướng (Time-Frequency Gated). | Khai thác sâu các đặc trưng tần số để theo dõi quỹ đạo suy thoái dài hạn. | Vận hành như một hộp đen hướng dữ liệu thuần túy; yêu cầu nhãn giám sát; không có cơ chế thiết lập ngưỡng chống rò rỉ. | Mô hình đề xuất xây dựng trên khung dự báo không giám sát và tích hợp khối POT chống rò rỉ dữ liệu trực tuyến. |
| PG-TMT (2026) [13] | Giám sát sức khỏe vòng bi trực tuyến | Tập dữ liệu run-to-failure của vòng bi công nghiệp | Bộ mã hóa ba nhánh (Tri-branch) kết hợp Tiny-Mamba, tích chập phân tách sâu và tri thức vật lý dải tần. | Kiến trúc gọn nhẹ phù hợp giám sát thời gian thực; tích hợp tri thức tần số vật lý. | Cấu trúc phân nhánh phức tạp; quá trình tính ngưỡng chẩn đoán vẫn phụ thuộc vào phân phối kiểm định toàn cục. | Mô hình đề xuất đơn giản hóa cấu trúc bằng chuỗi đơn phân rã xu hướng/mùa vụ và cô lập hiệu chuẩn tĩnh ở giai đoạn đầu. |
| EVT Open Set (2022) [17] | Chẩn đoán lỗi tập mở (Open Set) | Tập dữ liệu lỗi vòng bi cơ bản (CWRU) | Tích hợp Lý thuyết giá trị cực biên (EVT-POT) trực tiếp vào không gian ẩn học sâu. | Thiết lập biên quyết định toán học chính xác cho các lỗi chưa từng xuất hiện trong tập huấn luyện. | Quá trình hiệu chuẩn ngưỡng POT sử dụng dữ liệu validation toàn cục, gây rò rỉ thông tin tương lai. | Mô hình đề xuất áp dụng POT nhưng thực thi quy trình hiệu chuẩn cục bộ không rò rỉ (condition == 0). |
| Mô hình Đề xuất (Proposed) | Phát hiện bất thường vòng bi không giám sát | Các tập dữ liệu rung động suy thoái thế giới thực | Mô hình lai Mamba-CNN phân rã chuỗi, tích hợp Stats Head 8 chiều và hiệu chuẩn POT cục bộ. | Độ phức tạp tuyến tính $O(N)$; diễn giải cơ học rõ ràng; quy trình hiệu chuẩn không rò rỉ dữ liệu thực tế. | Hiệu năng phụ thuộc vào chất lượng của phân đoạn dữ liệu khỏe mạnh ở giai đoạn đầu. | (Thiết lập đường cơ sở khoa học mới cho bài toán PHM không rò rỉ dữ liệu). |

Methodology

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
