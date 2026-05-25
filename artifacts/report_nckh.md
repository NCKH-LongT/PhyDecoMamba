Multi Scale Series Decomposed State Space Model for High Precision Bearing Anomaly Detection

Truong Binh Thuan, Ly Hung Lam

Faculty of Information Technology

Ton Duc Thang University

Ho Chi Minh City, Vietnam

e-mail: sunbv56@gmail.com, lyhunglam2004@gmail.com

Truong Long

line 1 (of Affiliation): dept. name of organization

Amazing Total Solutions Joint Stock Company

Ho Chi Minh City, Vietnam

e-mail: admin@amazingtech.vn

Abstract: This paper proposed an advanced framework for bearing anomaly detection utilizing the Mamba state space model architecture. Anomaly detection from raw vibration signals often encountered significant challenges due to the non-stationary nature and complex noise of time series data. To address these issues, the proposed model integrated a Reversible Instance Normalization layer to mitigate distribution shifts, combined with a series decomposition technique to separate time series into trend and seasonal components. These decomposed components were subsequently processed through a Mamba encoder to effectively extract long-range dependencies. Forecasting and denormalization processes were sequentially executed to reconstruct the signals, from which the mean squared error was calculated to serve as the anomaly score. Finally, the framework employed the Peak Over Threshold dynamic thresholding technique to establish automated warning boundaries. Experimental results on real-world vibration datasets demonstrated that the proposed model achieved high precision and significantly reduced false alarm rates. The study concluded that the integration of series decomposition and state space models provided a reliable solution for industrial fault diagnosis.

Keywords: state space models; time series decomposition; bearing anomaly detection; fault diagnosis; Peak Over Threshold.

Introduction

Rolling element bearings serve as the "heart" of rotating machinery, yet they remain one of the most vulnerable components, accounting for approximately 45% to 70% of all electrical machine failures [1]. In the era of Industry 4.0, Prognostics and Health Management (PHM) has transitioned from reactive maintenance to proactive anomaly detection, leveraging high-frequency vibration data to identify incipient structural degradations before they escalate into catastrophic system failures. From a rotational dynamics perspective, when a microscopic defect develops on the inner race, outer race, or rolling elements, each mechanical interaction releases an extremely short-duration impulse. These periodic impacts excite the natural structural resonances of the machinery, producing complex vibration signals that can be captured by industrial accelerometers. To record these transient events, sensors must acquire measurements at exceptionally high sampling rates, often reaching tens of kilohertz. Therefore, early-stage anomaly detection within these streams is a core objective of modern PHM systems to prevent catastrophic breakdowns and minimize unplanned downtime.

To analyze these high-frequency vibration waveforms, various data-driven deep learning models have been proposed for automated feature extraction; however, these models often struggle with the "curse of long-range dependency" — where the signal length required to capture complete degradation cycles exceeds the effective receptive field of the network. Specifically, Long Short-Term Memory (LSTM) networks face a major computational bottleneck due to their sequential execution, which prevents parallel training on modern hardware and leads to vanishing gradients when modeling long sequences [2]. While Temporal Convolutional Networks (TCN) architectures alleviate sequential constraints by utilizing dilated causal convolutional filters to process inputs in parallel, their ability to capture temporal dependencies remains limited by a fixed receptive field [3]. Transformer-based models successfully capture global dependencies using self-attention mechanisms, yet their quadratic computational and memory complexity, scaled at , poses a significant bottleneck for real-time edge monitoring where resources are constrained [4]. Furthermore, because Transformers treat each time step as an isolated token, they lack local translation invariance, making them highly sensitive to industrial noise and prone to false alarms. Indeed, industrial data are collected under highly noisy conditions, which often obscure the subtle features of early faults.

The recent introduction of the Mamba architecture, a selective State-Space Model (SSM), has redefined the efficiency-accuracy trade-off by achieving linear complexity  [5]. However, existing Mamba-based PHM models, such as FEMamba [6] and TFG-Mamba [7], primarily focus on direct Remaining Useful Life (RUL) regression. These approaches often overlook two critical physical characteristics of bearing vibration: (i) the non-stationary nature of signal components in practical industrial environments, where operating variables such as motor speeds, loads, and temperatures fluctuate dynamically, causing the statistical properties of the collected vibration data to drift continuously; and (ii) the multi-scale nature of fault signatures, which range from broad energy shifts to micro-second impulse transients. When autoencoders are trained on stationary, healthy data, these distribution shifts degrade the reconstruction accuracy of the models, triggering false alerts. Furthermore, a critical research gap remains in the lack of unified frameworks that integrate multi-scale temporal decomposition within the network to decouple trend and seasonal components separately [9], combined with statistically rigorous dynamic thresholding techniques instead of rigid, empirical thresholds that are highly sensitive to background noise [10].

To bridge these gaps, this paper introduces HybridMamba++, a Channel-Independent (CI) architecture specifically engineered for bearing PHM. At its core, we propose a Hybrid Mamba-CNN framework that integrates the selective State-Space Model (SSM) with a Convolutional Neural Network (CNN). We choose this hybrid design because the CNN branch acts as a dynamic spatial-temporal filter to isolate high-frequency noise and capture impact-like micro-transients, while the Mamba branch tracks long-horizon degradation trends with linear computational complexity [5]. Consequently, the central research question of this investigation is: Can a unified framework that combines a hybrid Mamba-CNN encoder, Reversible Instance Normalization (RevIN) for distribution-shift mitigation [8], multi-scale series decomposition [9], and a Peak Over Threshold (POT) dynamic thresholding mechanism based on Extreme Value Theory (EVT) [10] achieve superior anomaly detection accuracy and lower false alarm rates compared to traditional deep learning models?

The main contributions of this work are four-fold:

Series-Decomposed Hybrid Architecture: We propose the architecture which leverages a hybrid Mamba-CNN configuration to capture both local transient impacts and global degradation dynamics with linear-time complexity [5].

Multi-Scale Dual-Kernel Patching: A novel front-end block that captures multi-frequency transients using asymmetric convolutional kernels, significantly improving the sensitivity to early-stage faults.

EVT-Based Threshold Calibration: Integration of the POT method [10] for automated, statistically-grounded anomaly detection by modeling the statistical tail of the reconstruction Mean Squared Error (MSE), achieving a near-zero False Alarm Rate (FAR) of 2.03%.

Rigorous Hardware-Aware Benchmarking: A comparative study under a strict 250k parameter budget, demonstrating that HybridMamba++ outperforms Transformers [4] and ModernTCN [11] in both precision (AUPRC: 0.9944) and inference speed (0.0673 ms/sample) under highly non-stationary conditions.

The remainder of this paper is organized as follows. Section II reviews foundational work in bearing PHM and literature on anomaly detection; Section III details the proposed CI-Mamba methodology and experimental setup; Section IV evaluates the experimental results and provides a comparative discussion; and Section V concludes the paper.

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
