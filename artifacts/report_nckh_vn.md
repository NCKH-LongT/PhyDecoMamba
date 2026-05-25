Multi Scale Series Decomposed State Space Model for High Precision Bearing Anomaly Detection

Truong Binh Thuan, Ly Hung Lam

Faculty of Information Technology

Ton Duc Thang University

Ho Chi Minh City, Vietnam

e-mail: sunbv56@gmail.com, lyhunglam2004@gmail.com

Truong Long

line 1 (of Affiliation): dept. name of organization

AMAZING TOTAL SOLUTIONS JOINT STOCK COMPANY

Ho Chi Minh City, Vietnam

e-mail: admin@amazingtech.vn

Abstract: Nghiên cứu này trình bày một phương pháp tiếp cận tiên tiến cho bài toán phát hiện bất thường của vòng bi dựa trên cấu trúc mô hình không gian trạng thái Mamba. Bài toán phát hiện bất thường từ tín hiệu rung thô thường đối mặt với thách thức lớn do tính chất không ổn định và nhiễu phức tạp của dữ liệu chuỗi thời gian. Để giải quyết vấn đề này, mô hình đề xuất đã tích hợp lớp chuẩn hóa RevIN nhằm giảm thiểu sự dịch chuyển phân phối, kết hợp với kỹ thuật phân tách chuỗi thời gian thành các thành phần xu hướng và mùa vụ. Các chuỗi thành phần sau đó được xử lý thông qua khối mã hóa Mamba để trích xuất hiệu quả các phụ thuộc xa. Quá trình dự báo và giải chuẩn hóa được thực hiện liên tiếp để tái tạo tín hiệu, từ đó sai số bình phương trung bình được tính toán đóng vai trò là điểm số bất thường. Cuối cùng, phương pháp sử dụng kỹ thuật vượt ngưỡng động POT để thiết lập ranh giới cảnh báo tự động. Kết quả thực nghiệm trên các tập dữ liệu tín hiệu rung thực tế đã chứng minh rằng mô hình đề xuất đạt độ chính xác cao và giảm thiểu đáng kể tỷ lệ báo động giả. Nghiên cứu kết luận rằng sự kết hợp giữa phân tách chuỗi và mô hình không gian trạng thái cung cấp một giải pháp chẩn đoán lỗi công nghiệp đáng tin cậy.

Keywords: mô hình không gian trạng thái; phân tách chuỗi thời gian; phát hiện bất thường vòng bi; chẩn đoán lỗi; ngưỡng động POT.

Introduction

Trong kỷ nguyên Cách mạng Công nghiệp lần thứ năm và sự chuyển dịch toàn cầu hướng tới các nhà máy thông minh, độ tin cậy vận hành và an toàn của máy móc quay đã trở thành yếu tố vô cùng quan trọng. Trong các hệ thống cơ khí này, vòng bi là bộ phận dễ bị tổn thương nhất, với các số liệu thống kê lịch sử chỉ ra rằng các lỗi liên quan đến vòng bi chiếm từ 45% đến 70% tổng số sự cố của máy móc điện quay [1]. Dưới góc độ động lực học quay, khi một khuyết tật vi mô xuất hiện trên rãnh trong, rãnh ngoài hoặc các con lăn, mỗi tương tác cơ học sẽ giải phóng một xung có thời gian tồn tại cực kỳ ngắn. Các tác động định kỳ này kích thích các tần số cộng hưởng cấu trúc tự nhiên của máy móc, tạo ra các tín hiệu rung phức tạp được ghi lại bởi các cảm biến gia tốc công nghiệp. Để ghi lại các sự kiện chuyển tiếp này, các cảm biến phải thu thập các phép đo ở tần suất lấy mẫu đặc biệt cao, thường đạt tới hàng chục kilohertz. Do đó, việc phát hiện bất thường ở giai đoạn sớm trong các luồng dữ liệu này là mục tiêu cốt lõi của các hệ thống quản lý sức khỏe và tiên lượng (Prognostics and Health Management - PHM) nhằm ngăn ngừa các sự cố thảm khốc và giảm thiểu thời gian dừng máy ngoài kế hoạch.

Để phân tích các dạng sóng rung động tần số cao này, nhiều mô hình học sâu dựa trên dữ liệu đã được đề xuất để tự động trích xuất đặc trưng; tuy nhiên, các kiến trúc truyền thống thường phải đối mặt với "lời nguyền phụ thuộc xa" — nơi chiều dài tín hiệu cần thiết để bắt trọn chu kỳ suy thoái vượt quá trường đón nhận hiệu dụng của mạng. Cụ thể, các mạng bộ nhớ dài ngắn (Long Short-Term Memory - LSTM) đối mặt với nút thắt cổ chai tính toán lớn do cơ chế thực thi tuần tự, điều này ngăn cản việc huấn luyện song song trên phần cứng hiện đại và dẫn đến hiện tượng tiêu biến đạo hàm khi mô hình hóa các chuỗi dài [2]. Mặc dù các kiến trúc mạng tích chập thời gian (Temporal Convolutional Network - TCN) giảm bớt các ràng buộc tuần tự bằng cách sử dụng các bộ lọc tích chập nhân quả giãn để xử lý song song, khả năng nắm bắt các phụ thuộc thời gian của chúng vẫn bị giới hạn bởi một trường đón nhận cố định [3]. Các mô hình dựa trên Transformer giải quyết vấn đề này bằng cách sử dụng cơ chế tự chú ý để nắm bắt các phụ thuộc toàn cục, nhưng độ phức tạp về tính toán và bộ nhớ bậc hai  của chúng khiến việc triển khai tại biên theo thời gian thực cho các tín hiệu cửa sổ dài trở nên không khả thi trong điều kiện tài nguyên hạn chế [4]. Hơn nữa, vì các mô hình Transformer xử lý mỗi bước thời gian như một mã báo độc lập, chúng thiếu tính bất biến tịnh tiến cục bộ, khiến chúng rất nhạy cảm với nhiễu công nghiệp và dễ đưa ra cảnh báo giả. Trên thực tế, dữ liệu công nghiệp được thu thập dưới các điều kiện có độ nhiễu cao, thường làm lu mờ các đặc trưng tinh vi của lỗi giai đoạn sớm.

Sự ra đời gần đây của kiến trúc Mamba, một mô hình không gian trạng thái chọn lọc (Selective State-Space Model - SSM), đã định nghĩa lại sự đánh đổi giữa hiệu năng và độ chính xác bằng cách đạt được độ phức tạp thời gian tuyến tính  [5]. Tuy nhiên, các mô hình PHM dựa trên Mamba hiện tại, chẳng hạn như FEMamba [6] và TFG-Mamba [7], chủ yếu tập trung vào bài toán hồi quy trực tiếp thời gian hữu dụng còn lại (Remaining Useful Life - RUL). Các cách tiếp cận này thường bỏ qua hai đặc tính vật lý quan trọng của hiện tượng rung động vòng bi: (i) tính chất phi tĩnh của các thành phần tín hiệu trong môi trường công nghiệp thực tế, nơi các biến số vận hành như tốc độ, tải trọng và nhiệt độ biến động liên tục làm dịch chuyển phân phối dữ liệu; và (ii) tính chất đa quy mô của các dấu hiệu lỗi, trải dài từ các dịch chuyển năng lượng rộng đến các xung chuyển tiếp vi giây. Khi các mô hình được huấn luyện trên dữ liệu tĩnh, các hiện tượng dịch chuyển phân phối này làm giảm đáng kể độ chính xác tái tạo, kích hoạt các cảnh báo giả. Hơn nữa, một khoảng trống nghiên cứu quan trọng vẫn tồn tại do thiếu các khung thống nhất tích hợp cơ chế phân tách chuỗi đa quy mô trực tiếp bên trong mạng để tách biệt thành phần xu hướng và mùa vụ [9], kết hợp với các kỹ thuật thiết lập ngưỡng động nghiêm ngặt về mặt thống kê thay vì các ngưỡng thực nghiệm cố định nhạy cảm với nhiễu nền [10].

Để thu hẹp những khoảng trống này, nghiên cứu này giới thiệu một kiến trúc độc lập kênh (Channel-Independent - CI) tiên tiến mang tên HybridMamba++ (Mô hình Không gian Trạng thái Phân tách Chuỗi Đa Quy mô) được thiết kế riêng cho bài toán PHM vòng bi. Tại lõi của mô hình, chúng tôi đề xuất một kiến trúc kết hợp Mamba-CNN tích hợp mô hình không gian trạng thái chọn lọc với một mạng tích chập thời gian. Nhánh CNN hoạt động như một bộ lọc không gian-thời gian động để cô lập nhiễu tần số cao và bắt các vi chuyển tiếp dạng xung, trong khi nhánh Mamba theo dõi các xu hướng suy thoái dài hạn với độ phức tạp tuyến tính [5]. Do đó, câu hỏi nghiên cứu trung tâm của cuộc điều tra này là: Liệu một khung thống nhất kết hợp bộ mã hóa lai Mamba-CNN, lớp chuẩn hóa đảo ngược đối xứng (Reversible Instance Normalization - RevIN) để giảm thiểu dịch chuyển phân phối [8], phân tách chuỗi đa quy mô [9], và cơ chế ngưỡng động vượt ngưỡng (Peak Over Threshold - POT) dựa trên lý thuyết giá trị cực biên (Extreme Value Theory - EVT) [10] có thể đạt được độ chính xác phát hiện bất thường vượt trội và tỷ lệ cảnh báo giả thấp hơn so với các mô hình học sâu truyền thống hay không?

Các đóng góp chính của công trình này gồm bốn phần:

Kiến trúc lai phân tách chuỗi: Chúng tôi đề xuất mô hình tận dụng cấu hình lai Mamba-CNN để bắt cả các tác động chuyển tiếp cục bộ và động lực học suy thoái toàn cục với độ phức tạp thời gian tuyến tính [5].

Vá đắp đa quy mô hai nhân: Một khối front-end mới giúp bắt các tiến trình chuyển tiếp đa tần số bằng cách sử dụng các nhân tích chập bất đối xứng, tăng cường độ nhạy với lỗi sớm.

Hiệu chuẩn ngưỡng dựa trên EVT: Tích hợp phương pháp POT [10] để thiết lập các ngưỡng động tự động hợp lý về mặt toán học, mô hình hóa phần đuôi thống kê của sai số tái tạo để đạt tỷ lệ cảnh báo giả (False Alarm Rate - FAR) gần bằng không là 2.03%.

Đánh giá nhận thức phần cứng nghiêm ngặt: Một nghiên cứu so sánh dưới ngân sách tham số nghiêm ngặt 250k, chứng minh rằng HybridMamba++ vượt trội hơn Transformer [4] và ModernTCN [11] về cả độ chính xác (AUPRC: 0.9944) và tốc độ suy luận (0.0673 ms/mẫu) dưới các điều kiện phi tĩnh cao.

Phần còn lại của bài viết được tổ chức như sau. Mục II xem xét các tài liệu liên quan về PHM vòng bi; Mục III chi tiết hóa phương pháp kiến trúc CI-Mamba đề xuất và thiết lập thực nghiệm; Mục IV đánh giá kết quả thực nghiệm và thảo luận so sánh; và Mục V kết luận bài viết.

literature review

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
