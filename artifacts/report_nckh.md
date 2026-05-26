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

Abstract: A hybrid Mamba-CNN architecture with a physics-informed statistical head is proposed for rolling element bearing anomaly detection under non-stationary operating conditions. Traditional deep sequence networks often struggle to isolate structural degradation patterns due to heavy industrial background noise and execution bottlenecks over long-horizon trajectories. To address these limitations, a distribution-adaptive framework utilizing temporal series decomposition is introduced to isolate trend and seasonal components from raw vibration signals. These temporal representations are subsequently enriched by an eight-dimensional physical statistical layer extracted from the lookback window, incorporating crucial diagnostic indicators including root mean square and kurtosis to guarantee mechanical interpretability. The integrated feature streams are processed through a channel-independent Mamba encoder to effectively model long-range temporal dependencies with linear complexity. Signal reconstruction is executed via sequential forecasting and denormalization processes, where the mean squared error is utilized as the anomaly score. To ensure a leakage-free calibration workflow aligned with true industrial deployment, localized decision thresholds are established per-bearing using the peak over threshold technique applied strictly to early-stage healthy operation segments. Under the evaluated settings, benchmark baselines—including long short-term memory, ModernTCN, PatchTST, and SimpleMamba—are auto-scaled to equalize the parameter budget. The empirical results indicate that enhanced structural efficiency, elevated detection accuracy, and reduced false alarm rates are achieved across real-world vibration datasets.

Keywords: selective state space models, hybrid Mamba-CNN, series decomposition, bearing anomaly detection, leakage-free calibration, peak over threshold.

Introduction

Reliability and operational safety in modern industrial asset management depend heavily on the continuous health monitoring of rotating machinery [14]. Within these mechanical systems, rolling element bearings are classified as critical components, as their structural degradation directly induces catastrophic system failures and substantial economic losses [15]. Consequently, vibration-based unsupervised anomaly detection has emerged as a cornerstone of Prognostics and Health Management (PHM) frameworks. Raw vibration signals collected from operational environments exhibit highly non-stationary distributions, severe non-linear dynamics, and are continuously corrupted by heavy industrial background noise. Although traditional signal processing techniques provide foundational diagnostic capabilities, subtle acoustic and transient impact pulses generated during early-stage degradation phases are frequently masked by operational noise, necessitating advanced sequence modeling paradigms capable of isolating anomalies with high structural fidelity.

In recent years, deep learning methodologies have been adopted to automate time-series feature extraction and health state classification [14]. Recurrent and temporal convolutional networks have been applied to sequence tracking, yet recursive update bottlenecks and fixed receptive fields inherently limit their effectiveness over long-horizon trajectories. To capture long-range interactions, advanced SOTA architectures—including Anomaly Transformer [5] and TimesNet [6]—leverage self-attention mechanisms and multi-frequency analysis. However, these frameworks suffer from quadratic computational complexity () and are prone to overfitting when subjected to industrial noise. Furthermore, alternative advanced baselines like ModernTCN [2] and PatchTST [7] segment signals into localized sub-series or patches to optimize processing efficiency. Despite these structural improvements, these methods operate primarily under channel-independent constraints, failing to effectively fuse cross-channel correlations with global temporal context, which leads to escalated false alarm rates in multi-sensor industrial setups.

More recently, selective state space models, notably the Mamba architecture [16], have been introduced to establish linear-time complexity () while maintaining a powerful selective scan mechanism. Specialized adaptations—including FEMamba [8] and TFG-Mamba [9]—attempt to incorporate time-frequency domain features to enhance diagnostic accuracy. Nevertheless, a critical research gap remains unaddressed within these state-of-the-art frameworks. First, existing Mamba-based diagnostic networks treat signal forecasting and thresholding as decoupled processes, ignoring the inherent distribution shifts of non-stationary signals. Second, pure data-driven sequence networks operate as uninterpretable black boxes, decoupling statistical anomaly scores from underlying mechanical degradation physics. Third, existing thresholding mechanisms frequently induce data leakage by utilizing global or future anomalous validation data during the calibration phase, rendering them impractical for true online industrial deployment workflows where only localized, historical healthy baselines are available.

To resolve these interconnected challenges, a physics-informed, series-decomposed hybrid Mamba-CNN framework is proposed in this study for leakage-free bearing anomaly detection. The core architecture integrates a temporal series decomposition module with moving average pooling to isolate low-frequency degradation trends from high-frequency seasonal shock transients. The seasonal feature streams are subsequently processed through a channel-independent Mamba encoder to maximize global dependency extraction without cross-channel noise propagation. To inject engineering domain knowledge and guarantee mechanical interpretability, the latent representations are reinforced by an eight-dimensional physical statistical layer (Stats Head) extracted directly from the lookback window. This layer incorporates Mean, Standard Deviation, Root Mean Square, Peak-to-Peak, Skewness, Kurtosis, Crest Factor, and Shape Factor, fusing raw data-driven representations with established mechanical profiles within a dedicated fusion forecasting head.

The main contributions of this research are summarized as follows. First, a hybrid Mamba-CNN forecasting paradigm with series decomposition is introduced, achieving linear complexity while eliminating cross-channel noise interference. Second, a physics-informed statistical head is formulated to bridge the gap between data-driven latent spaces and mechanical descriptors, ensuring structural interpretability. Third, a leakage-free calibration workflow utilizing Extreme Value Theory via the Peak Over Threshold (POT) technique is executed, where dynamic boundaries are derived strictly from localized, early-stage healthy operational segments. The remainder of this paper is organized as follows: Section II reviews the relevant literature and deep sequence baselines. Section III details the mathematical formulation of the proposed methodology. Section IV outlines the experimental setup, baseline parameter budgeting, and comparative performance analysis. Section V concludes the study and discusses future research trajectories.

Related Work

Deep Sequence Modeling in Prognostics and Health Management (PHM)

Empirical statistical evidence indicates that rolling element bearing failures account for approximately 40%–50% of all structural damages in rotating machinery, directly causing unplanned downtime and substantial economic deficits [15]. Developing automated vibration-based diagnostic monitoring systems within heavy industrial environments constantly encounters critical challenges due to non-stationary distributions, highly non-linear dynamics, and severe background noise contamination [14]. To extract sparse time-frequency domain features in an unsupervised manner, learnable Wavelet filter optimization frameworks—such as the DeSpaWN architecture [1]—have demonstrated foundational signal processing efficacy.

However, when migrating to long-sequence modeling tasks, traditional deep learning architectures exhibit inherent structural bottlenecks. Long Short-Term Memory (LSTM) networks rely on recursive hidden state updates, imposing severe sequential computational bottlenecks that preclude parallelization and induce vanishing gradients over extended horizons. Although Temporal Convolutional Networks (TCNs) mitigate parallel processing constraints via dilated convolutions, the receptive field remains inherently bounded by fixed filter kernel configurations, failing to capture long-horizon degradation trajectories when sensor sampling frequencies are high.

While modernized convolutional architectures like ModernTCN [2] optimize representation capacity via depth-wise and point-wise decompositions, critical analytical revisions such as ModernTCN Revisited [3] have exposed flaws regarding the "Drop Last Trick" and validation-stage data leakage within the original benchmarks. This underscores the imperative for rigorous, transparent experimental protocols and balanced parameter budgeting. Furthermore, conventional unsupervised Autoencoders applied to anomaly detection often suffer from optimization mismatches between training and inference phases, a limitation partially addressed by the joint semantic association mechanism in TimeRCD [4]. This study synthesizes the paradigms of [3] and [4] by implementing a sequence forecasting framework rather than crude reconstruction, while strictly enforcing an automated baseline parameter scaling protocol.

Transformer-based Time-series Models

The evolution of self-attention mechanisms within Transformer structures has advanced the capture of long-term correlations in time-series data. For unsupervised anomaly detection, the Anomaly Transformer architecture [5] established a foundation by measuring the association discrepancy between localized and global attention distributions to isolate anomalous instances. To thoroughly address the non-stationary nature of industrial signals, the TimesNet model [6] introduced an algorithmic transformation that converts one-dimensional (1D) time-series into two-dimensional (2D) structures via Fast Fourier Transform (FFT) multi-periodicity analysis, enabling the simultaneous exploitation of intra-period and inter-period variations. Subsequently, the PatchTST model [7] optimized Transformer efficiency by introducing a temporal patching mechanism combined with channel-independence (CI) constraints, successfully reducing the attention network's computational complexity to linear scale while mitigating cross-channel noise propagation.

Despite these foundational contributions, Transformer-based models still exhibit core bottlenecks when deployed in practical PHM domains. First, the quadratic computational and memory complexity () of the original self-attention matrix imposes severe hardware resource constraints when processing extended historical lookback windows. Second, and most critically, Transformer encoders lack localized spatial-temporal inductive biases and translation invariance. When vibration signals are corrupted by heavy industrial background noise or transient shocks from neighboring machinery, Transformers are highly susceptible to misinterpreting random noise peaks as structural degradation indicators, leading to escalated false alarm rates. The hybrid Mamba-CNN architecture proposed in this study resolves these deficiencies by fusing a linear-time selective scan mechanism with localized convolutional branches to maintain robustness against industrial noise.

Selective State Space Models (Mamba) for PHM

To bypass the computational limitations of Transformers, the foundational selective state space modeling framework (Mamba) introduced by Gu and Dao [16] established a new milestone, achieving linear-time complexity () via an input-dependent selective scan mechanism. In the PHM domain and bearing fault diagnosis, Mamba-based variants have emerged extensively during the 2025–2026 period. The FEMamba architecture [8] integrates an adaptive MFES block and a space-constrained DSGR module for supervised remaining useful life (RUL) estimation. The TFG-Mamba framework [9] executes time-frequency domain fusion through a guided gated Mamba mechanism to capture long-term degradation trends in rotating machinery.

For complex multivariate sequences, the TimeMachine model [10] deploys a Quadruple-Mamba architecture to simultaneously integrate channel-mixing and channel-independence, whereas the TSC-Mamba architecture [11] applies a "Decomposition-Propagation-Cross-Correlation" pipeline via a low-rank CIFM module to optimize sequence representation. Aiming for practical implementation, the LDGM (Mamba-MP) model [12] supports the separation of global dynamic features and localized micro-impulses for intelligent edge deployment. Notably, the PG-TMT (Physics-Guided Tiny-Mamba Transformer) framework [13] in IEEE Transactions on Reliability demonstrated the feasibility of combining compact deep sequence networks with physical frequency band knowledge for real-time online monitoring.

Nevertheless, an honest limitation across these state-of-the-art Mamba-based studies is that they are predominantly structured as purely data-driven "black-box" models, operating either under supervised fault classification or RUL regression paradigms. In actual industrial scenarios, labeled fault data is highly scarce and expensive to acquire. These frameworks completely lack an unsupervised sequence forecasting mechanism constrained and directly guided by an explicit physical statistical block. The present study addresses this research gap by embedding an 8-dimensional physical Stats Head to guide Mamba’s latent space, establishing a practical unsupervised forecasting paradigm.

Dynamic Thresholding and Anomaly Calibration Methods

A complete anomaly detection framework depends not only on the network architecture but is ultimately determined by its decision threshold calibration mechanism. Traditional static thresholds, such as standard 3-Sigma limits or robust Interquartile Range (IQR) intervals, fail under variable operating conditions (e.g., changing speeds and loads) due to continuous distribution shifts in the target domain. Dynamic statistical models like Gaussian Mixture Models (GMM) have been applied to model decision boundaries, yet they often lack a rigorous mathematical foundation when encountering extreme out-of-distribution values.

Integrating Extreme Value Theory (EVT) via the Peak Over Threshold (POT) technique directly into the latent spaces of deep learning models, as explored in [17], has demonstrated the capability to establish precise open-set decision boundaries and systematically isolate anomalies. Regardless of these benefits, current calibration frameworks in the PHM literature frequently induce severe, inadvertent data leakage. This error occurs because threshold hyperparameter optimization is executed over the entire validation distribution (including future early-stage degradation or complete failure segments), rendering the model incapable of true online deployment on physical assets—where the system only has access to localized historical healthy operational data. The proposed model completely eliminates this vulnerability via a strict localized calibration workflow, computing the POT configuration solely from early-stage healthy data segments, where the operational status indicators are flagged as healthy (condition variable == 0).

Literature Comparison Matrix Table

TABLE I COMPREHENSIVE LITERATURE MATRIX AND STRUCTURAL COMPARISON WITH THE PROPOSED PARADIGM

| Study | Task | Dataset | Method | Strength | Limitation | Relation to this work |
| --- | --- | --- | --- | --- | --- | --- |
| Anomaly Transformer (2022) [5] | Unsupervised Anomaly Detection | Multi-domain time-series (SMD, MSD, etc.) | Association Discrepancy computation mechanism utilizing specialized Transformers. | Effectively isolates cyclical anomalies via localized and global attention distributions. | Suffers from quadratic computational complexity (); highly sensitive to industrial noise. | The proposed model replaces the self-attention block with a Mamba encoder to achieve linear complexity (). |
| TimesNet (2023) [6] | Forecasting and Anomaly Detection | Non-stationary time-series (Weather, ETT, etc.) | Transforms 1D series into 2D structures via FFT multi-periodicity analysis and Inception blocks. | Effectively captures non-stationary properties through simultaneous multi-period analysis. | High computational overhead due to 2D image processing; pure CNN lacks selective long-sequence scanning. | The proposed framework implements temporal series decomposition directly in the time domain to avoid FFT overhead. |
| PatchTST (2023) [7] | Long-term Time-series Forecasting | Standard benchmark time-series (Electricity, Traffic) | Temporal patching technique combined with channel-independence constraints. | Reduces Transformer complexity to linear scale; eliminates cross-channel noise propagation. | Ignores cross-channel correlations, degrading detection performance in multi-sensor setups. | The proposed model maintains channel independence in the Mamba branch but fuses channels within the Fusion Head. |
| TFG-Mamba (2026) [9] | Machinery Degradation Trend Forecasting | Real-world run-to-failure bearing datasets | Time-frequency domain fusion via a guided gated Mamba mechanism. | Deeply exploits frequency-domain descriptors to track long-term degradation trajectories. | Operates as a purely data-driven black box; requires supervised labels; lacks leakage-free thresholding. | The proposed model builds upon an unsupervised forecasting framework and integrates a leakage-free POT module. |
| PG-TMT (2026) [13] | Online Bearing Health Monitoring | Industrial run-to-failure bearing datasets | Tri-branch encoder combining Tiny-Mamba, deep separable convolutions, and frequency-band physical knowledge. | Lightweight architecture suitable for real-time edge monitoring; incorporates physical diagnostics. | Highly complex branching structure; threshold calibration still relies on global validation distributions. | The proposed model simplifies the structure using a single branch with trend-seasonal decomposition and localized calibration. |
| EVT Open Set (2022) [17] | Open-set Fault Diagnosis | Fundamental bearing datasets (CWRU) | Embedding Extreme Value Theory (EVT-POT) directly into deep learning latent spaces. | Establishes precise mathematical boundaries for unseen fault types during training. | The POT threshold calibration process utilizes global validation sequences, causing future data leakage. | The proposed architecture applies POT but strictly executes a localized calibration workflow (condition == 0). |
| Proposed Model | Unsupervised Bearing Anomaly Detection | Real-world industrial run-to-failure vibration datasets | Hybrid Mamba-CNN with series decomposition, an 8-D physical Stats Head, and localized POT calibration. | Achieves linear complexity $O(N)$; guarantees mechanical interpretability; provides leakage-free online calibration. | Performance depends on the structural integrity of the early-stage healthy data segments. | (Establishes a new scientific baseline for leakage-free, physics-informed PHM frameworks). |

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
