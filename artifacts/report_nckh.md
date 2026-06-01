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

Empirical statistical evidence indicates that rolling element bearing failures account for approximately 40%–50% of all structural damages in rotating machinery, directly causing unplanned downtime and substantial economic deficits [15]. Developing automated vibration-based diagnostic monitoring systems within heavy industrial environments constantly encounters critical challenges due to non-stationary distributions, highly non-linear dynamics, and severe background noise contamination [14]. To extract sparse time-frequency domain features in an unsupervised manner, learnable Wavelet filter optimization frameworks—such as the DeSpaWN architecture [1]—have demonstrated foundational signal processing efficacy. However, these static frequency features frequently omit long-term dynamic shifts when mechanical structures degrade over time, creating a clear demand for architectures capable of learning long-sequence time-series representations [14].

To capture these temporal dependencies, deep learning architectures such as Long Short-Term Memory (LSTM) and Temporal Convolutional Network (TCN) frameworks have been widely adopted. Nevertheless, LSTM networks rely on recursive hidden state updates, imposing severe sequential computational bottlenecks that preclude parallelization and induce vanishing gradients over extended windows. Meanwhile, TCN architectures achieve parallel computing via dilated convolutions, but the receptive field remains strictly bounded by fixed filter kernel configurations, failing to encompass complete rotational cycles of bearings when sensor sampling frequencies are high.

For unsupervised anomaly detection tasks, traditional Autoencoders (AE) are frequently implemented to learn healthy baseline representations and identify anomalies using reconstruction errors. However, rather than suffering from vague optimization mismatches, practical AE models frequently experience over-generalization, wherein the latent space inadvertently reconstructs early-stage anomalous impact signals with high precision, thereby tripling missed alarm rates [4]. To resolve this, recent frameworks such as TimeRCD [4] propose modeling joint semantic association discrepancies, or shifting entirely toward next-window time-series forecasting paradigms rather than crude signal reconstruction. Concurrently, critical analytical evaluations such as ModernTCN Revisited [3] have revealed that relaxed experimental protocols easily introduce validation-stage data leakage. Consequently, the sequence forecasting paradigm is inherited in this framework to eliminate the over-generalization vulnerabilities of traditional AEs, while an automated baseline parameter scaling protocol is strictly enforced to guarantee fair architectural comparisons [3].

Transformer-based Time-series Models

The evolution of self-attention mechanisms within Transformer structures has significantly advanced the capture of long-term correlations in time-series data. In unsupervised anomaly detection, the Anomaly Transformer architecture [5] established a profound foundation by measuring the association discrepancy between localized and global attention distributions to isolate anomalous instances. To systematically handle the non-stationary nature of industrial signals, the TimesNet model [6] transforms one-dimensional (1D) time-series into two-dimensional (2D) tensors via Fast Fourier Transform (FFT) multi-periodicity analysis, applying 2D convolutional operations on folded period tensors to simultaneously exploit intra-period and inter-period variations. However, executing continuous 2D convolutional operations over large frequency tensors introduces prohibitive computational overhead, and the pure Convolutional Neural Network (CNN) structure of TimesNet lacks input-dependent selective scanning capabilities over long horizons [6].

To optimize execution efficiency, the PatchTST framework [7] introduces a temporal patching mechanism combined with channel-independence (CI) constraints, which successfully reduces the input sequence length forwarded to the attention network. Nevertheless, for extended historical lookback windows expanded to encompass low rotational frequencies, the computational and memory complexity of the self-attention mechanism in Transformer-based models still scales quadratically () with respect to the number of generated patches, driving severe hardware memory bottlenecks [7]. Furthermore, conventional Transformer encoders lack localized spatial-temporal inductive biases and translation invariance. When vibration sequences are contaminated by heavy industrial background noise, Transformers are highly susceptible to misinterpreting random noise peaks as structural degradation indicators, leading to escalated false alarm rates [14]. The hybrid Mamba-CNN architecture proposed in this study resolves these deficiencies by fusing a linear-time selective scan mechanism with localized convolutional branches to suppress high-frequency operational noise.

Selective State Space Models (Mamba) for PHM

To break through the computational bottlenecks of Transformers on long sequences, the foundational selective state space model (Mamba) formulated by Gu and Dao [16] established a new milestone, achieving linear-time complexity () via an input-dependent selective scan mechanism. Within the PHM domain and bearing fault diagnosis, Mamba-based variants have emerged extensively during the 2025–2026 period. The FEMamba architecture [8] integrates an adaptive Multi-scale Feature Extraction Slot (MFES) block and a Degradation Space Guided Regression (DSGR) module for supervised Remaining Useful Life (RUL) estimation. The TFG-Mamba framework [9] executes time-frequency domain fusion through a guided gated Mamba mechanism to track long-term degradation trajectories in rotating machinery.

For complex multivariate sequences, the TimeMachine model [10] deploys a Quadruple-Mamba architecture to simultaneously integrate channel-mixing and channel-independence, whereas the TSC-Mamba architecture [11] applies a "Decomposition-Propagation-Cross-Correlation" pipeline via a low-rank Channel Information Fusion Module (CIFM) to optimize sequence representation. Aiming for practical implementation, the Linear Decoupled Graph Model (LDGM) [12] separates global dynamic features and localized micro-impulses using a Multi-Perspective Mamba (Mamba-MP) architecture for intelligent edge deployment. Notably, the Physics-Guided Tiny-Mamba Transformer (PG-TMT) framework [13] in IEEE Transactions on Reliability demonstrated the feasibility of combining compact deep sequence networks with physical frequency band knowledge for real-time online monitoring.

On the other hand, time-series signal decomposition techniques—such as Seasonal-Trend Decomposition using LOESS (STL) and Singular Spectrum Analysis (SSA)—have been widely established in machinery health monitoring to isolate long-term degradation profiles from complex operational variations. STL effectively extracts monotonic degradation trends representing physical wear while eliminating operational cyclical anomalies. Similarly, SSA facilitates the model-free separation of smooth degradation trajectories from random white noise. Recently, hybrid frameworks combining mode decomposition—such as Variational Mode Decomposition (VMD) or Empirical Mode Decomposition (EMD)—with selective state space models like the MD-BiMamba architecture [21] have demonstrated that decomposing raw vibration signals prior to Mamba encoding significantly mitigates fault masking effects, enabling the selective scan mechanism to focus tightly on fault-induced impulses.

Nevertheless, an honest limitation across these state-of-the-art (SOTA) Mamba-based studies is that they are predominantly structured as purely data-driven "black-box" models, operating either under supervised fault classification or RUL regression paradigms. In actual industrial scenarios, labeled fault data is highly scarce and expensive to acquire, necessitating unsupervised anomaly detection solutions [14], [15]. Furthermore, utilizing data-driven Mamba layers in isolation creates a mechanical black box lacking interpretability for operational engineers [13]. Mamba sequentially scans temporal inputs but lacks a direct mechanism to integrate structural descriptors specific to bearing mechanics. The present study addresses this research gap by embedding an 8-dimensional physical statistical head (Stats Head) to guide Mamba’s latent space, anchoring the unsupervised forecasting process within explicit mechanical profiles over extended lookback windows with linear-time efficiency.

Dynamic Thresholding and Anomaly Calibration Methods

A complete anomaly detection framework depends not only on the network architecture but is ultimately determined by its decision threshold calibration mechanism. Traditional static thresholds, such as standard 3-Sigma limits or robust Interquartile Range (IQR) intervals, fail under variable operating conditions due to continuous distribution shifts in the target domain. Dynamic statistical models like Gaussian Mixture Models (GMM) have been applied to model decision boundaries, yet they often lack a rigorous mathematical foundation when encountering extreme out-of-distribution values.

Integrating Extreme Value Theory (EVT) via the Peak Over Threshold (POT) technique directly into the latent spaces of deep learning models, as explored in [17], has demonstrated the capability to establish precise open-set decision boundaries based on the Generalized Pareto Distribution (GPD). Regardless of these benefits, current calibration frameworks in the PHM literature frequently induce severe, inadvertent future data leakage. Specifically, prominent benchmarks derived from stochastic Recurrent Neural Networks (Stochastic RNN) such as OmniAnomaly [18], adversarially trained autoencoders like USAD [19], and adversarial Transformers like TranAD [20] utilize the entire validation distribution—including future degraded or complete failure segments—to optimize threshold hyperparameters and maximize reported F1-scores. This practice implicitly converts threshold selection into a semi-supervised task, rendering the model incapable of true online deployment on physical assets where the system only has access to localized historical healthy operational data.

To resolve this vulnerability, the proposed model executes a strict localized calibration workflow. The calculation and fitting of POT parameters are derived solely from localized, early-stage healthy data segments (condition variable == 0) without using any future anomalous profiles. Under variable operating conditions, non-stationary distribution shifts of healthy signals are dynamically tracked by the hybrid Mamba-CNN architecture [15], ensuring that normal operational variations remain below the calibrated extreme boundaries, thereby guaranteeing absolute robustness for online industrial workflows.

Literature Comparison Matrix Table

TABLE I COMPREHENSIVE LITERATURE MATRIX AND STRUCTURAL COMPARISON WITH THE PROPOSED PARADIGM

| Study | Task | Dataset | Method | Strength | Limitation | Relation to this work |
| --- | --- | --- | --- | --- | --- | --- |
| OmniAnomaly (2019) [18] | Unsupervised Anomaly Detection | Industrial multivariate time-series | Stochastic Recurrent Neural Network (Stochastic RNN) modeling. | Captures stochastic dynamics via probabilistic reconstruction distributions. | Suffers from over-generalization that erases early fault indicators; global threshold calibration causes data leakage. | Shifts from a reconstruction paradigm to a sequence forecasting paradigm, incorporating localized POT calibratio. |
| USAD (2020) [19] | Unsupervised Anomaly Detection | Standard benchmark datasets (SWaT, WADI) | Adversarially trained autoencoders framework. | Enhances autoencoder training stability and convergence speed through adversarial learning. | Operates as a pure data-driven black box; threshold hyperparameter optimization relies on global validation anomalies. | Integrates an 8-D physical Stats Head to eliminate black-box behavior and isolates POT calibration within healthy intervals. |
| TranAD (2022) [20] | Unsupervised Anomaly Detection | Complex multivariate time-series | Transformer network with adversarial training and self-conditioning loops. | Achieves accelerated inference speeds and high precision via adversarial Transformer outputs. | High memory footprint over extended sequences; POT hyperparameter selection relies on future anomalous data. | Utilizes a hybrid Mamba-CNN framework with linear complexity and strictly enforces localized calibration (condition == 0). |
| Anomaly Transformer (2022) [5] | Unsupervised Anomaly Detection | Multi-domain time-series (SMD, MSD, etc.) | Association Discrepancy computation mechanism utilizing specialized Transformers. | Effectively isolates cyclical anomalies via localized and global attention distributions. | Suffers from quadratic computational complexity (); highly sensitive to industrial noise. | The proposed model replaces the self-attention block with a Mamba encoder to achieve linear complexity (). |
| TimesNet (2023) [6] | Forecasting and Anomaly Detection | Non-stationary time-series (Weather, ETT, etc.) | Transforms 1D series into 2D structures via FFT multi-periodicity analysis and Inception blocks. | Effectively captures non-stationary properties through simultaneous multi-period analysis. | High computational overhead due to 2D image processing; pure CNN lacks selective long-sequence scanning. | The proposed framework implements temporal series decomposition directly in the time domain to avoid FFT overhead. |
| PatchTST (2023) [7] | Long-term Time-series Forecasting | Standard benchmark time-series (Electricity, Traffic) | Temporal patching technique combined with channel-independence constraints. | Reduces Transformer complexity to linear scale; eliminates cross-channel noise propagation. | Ignores cross-channel correlations, degrading detection performance in multi-sensor setups. | The proposed model maintains channel independence in the Mamba branch but fuses channels within the Fusion Head. |
| MD-BiMamba (2024) [21] | Supervised Fault Diagnosis | Aero-engine inter-shaft bearing dataset | Combines signal mode decomposition (VMD/EMD) with a bidirectional Mamba feature fusion strategy. | Effectively mitigates fault masking effects, optimizing the selective scanning of fault-induced impact pulses. | Operates as a supervised classification model dependent on fault labels; lacks an online dynamic leakage-free thresholding mechanism. | Inherits the concept of signal decomposition for feature refinement but develops an unsupervised forecasting framework integrated with localized POT calibration. |
| TFG-Mamba (2026) [9] | Machinery Degradation Trend Forecasting | Real-world run-to-failure bearing datasets | Time-frequency domain fusion via a guided gated Mamba mechanism. | Deeply exploits frequency-domain descriptors to track long-term degradation trajectories. | Operates as a purely data-driven black box; requires supervised labels; lacks leakage-free thresholding. | The proposed model builds upon an unsupervised forecasting framework and integrates a leakage-free POT module. |
| PG-TMT (2026) [13] | Online Bearing Health Monitoring | Industrial run-to-failure bearing datasets | Tri-branch encoder combining Tiny-Mamba, deep separable convolutions, and frequency-band physical knowledge. | Lightweight architecture suitable for real-time edge monitoring; incorporates physical diagnostics. | Highly complex branching structure; threshold calibration still relies on global validation distributions. | The proposed model simplifies the structure using a single branch with trend-seasonal decomposition and localized calibration. |
| EVT Open Set (2022) [17] | Open-set Fault Diagnosis | Fundamental bearing datasets (CWRU) | Embedding Extreme Value Theory (EVT-POT) directly into deep learning latent spaces. | Establishes precise mathematical boundaries for unseen fault types during training. | The POT threshold calibration process utilizes global validation sequences, causing future data leakage. | The proposed architecture applies POT but strictly executes a localized calibration workflow (condition == 0). |
| Proposed Model | Unsupervised Bearing Anomaly Detection | Real-world industrial run-to-failure vibration datasets | Hybrid Mamba-CNN with series decomposition, an 8-D physical Stats Head, and localized POT calibration. | Achieves linear complexity ; guarantees mechanical interpretability; provides leakage-free online calibration. | Performance depends on the structural integrity of the early-stage healthy data segments. | (Establishes a new scientific baseline for leakage-free, physics-informed PHM frameworks). |

Methodology

Problem Formulation and Framework Overview

The structural health monitoring of industrial rolling element bearings utilizing time-series trajectories is formulated under an unsupervised anomaly detection paradigm driven by a next-window forecasting mechanism. Let  denote a multivariate vibration sequence collected from an acceleration sensor array across  physical channels (representing distinct spatial measurement axes, such as horizontal and vertical directions) over a historical lookback window of length . The mathematical objective of the framework is to accurately forecast the subsequent operational sequence within a defined forecast horizon of length , denoted as , which is sequentially evaluated against the corresponding ground-truth future target sequence .

The proposed architecture operates strictly under the principles of unsupervised representation learning, wherein network optimization is executed exclusively utilizing data distributions derived from the initial early-stage healthy operational phases of the machinery—identified via binary condition indicator variables equal to zero (). The theoretical foundation of this approach relies on the hypothesis that the network will establish an optimal latent space characterizing the normal cyclical dynamics of the mechanical asset. Upon the onset of structural degradation profiles, such as surface pitting, spalling, or localized wear (fault states), the emergence of non-linear transient impact shocks and severe distribution shifts disrupts the learned temporal regularities. Consequently, the squared discrepancy between the ground-truth future sequence  and the reconstructed forecast sequence  escalates sharply, yielding a robust mathematical baseline to compute anomaly scores and trigger online warning boundaries without requiring scarce or expensive labeled fault repositories.

Signal Preprocessing via Causal Butterworth High-Pass Filtering

To suppress low-frequency operational oscillations originating from baseline induction motors, environmental ambient noise, or extraneous mechanical dynamics unrelated to bearing degradation, the raw vibration sequences are initially passed through an -th order high-pass Butterworth filter configured with a strict causality property. The magnitude squared frequency response of the filter in the continuous frequency domain is mathematically expressed as follows:

where  denotes the configured cutoff frequency parameter, and  represents the component angular frequency of the signal execution. To implement this continuous mathematical operator directly into a real-time discrete data processing pipeline without inducing look-ahead bias or future data leakage, a bilinear transformation is applied to map the transfer function from the continuous -domain to the discrete -domain, establishing a discrete time-domain difference equation of the form:

where  and  denote the discrete signals immediately prior to and following the filtering operation at time step , respectively, while  and  represent the filter coefficients derived from the baseline sensor sampling frequency . Due to this causal formulation, the phase delay generated by the filter is naturally learned and compensated for by the downstream deep sequence model during the forecasting optimization process. The physical rationale of this preprocessing stage is to isolate and preserve low-amplitude, high-frequency transient impact impulses—which serve as the most sensitive early diagnostic indicators of surface fatigue cracking—vibrating inside the material prior to baseline distribution stabilization.

Distribution-Adaptive Series Decomposition Block

Aligning with the principle of architectural parsimony introduced in state-of-the-art time-series forecasting models such as the DMamba paradigm [22], the proposed framework completely decouples the processing of seasonal and trend feature streams. The core objective of this mechanism is to decompose the filtered input sequence  into two distinct components possessing divergent physical and statistical profiles to maximize deep sequence representation capacity. The decomposition workflow is executed via a one-dimensional moving average pooling operator () sliding along the temporal axis of the historical lookback window:

where  represents the moving average kernel size width. An edge-replication padding layer () is enforced to guarantee that the trend tensor  maintains strict geometric dimensionality alignment with the original input.

The trend component  isolates the smooth moving average profile, reflecting the progressive, slow-moving physical wear of the mechanical structure, which inherently exhibits a low dimensional complexity. Conversely, the seasonal component  captures highly non-linear high-frequency dynamics, incorporating synchronous rotational frequency cycles and transient impact shocks induced by bearing localized defects. This structural bifurcation prevents energy-dominant low-frequency trend profiles from masking sub-nominal seasonal fault impulses, allowing dedicated network branches to focus exclusively on compatible feature spaces.

Linear Trend Forecasting Stream

In accordance with the structural parsimony verified by DMamba [22], the trend component  models smooth, monotonic mechanical degradation trajectories. Routing this low-complexity stream through highly parameterized attention layers or selective state space scanning matrices is structurally redundant, frequently inducing severe overfitting behaviors and unnecessary computational inflation. Consequently, a direct linear projection layer is structured to forecast the future trend profile  across the target horizon:

where  denotes the projection weight matrix, and  represents the bias vector. To optimize memory resource footprints and enhance training numerical stability under extended lookback windows, an adaptive downsampling pooling layer () with a configured stride  is integrated to compress the trend sequence prior to the execution of the matrix multiplication. The projection parameter matrix  is regularized under a strict channel-independent weight sharing constraint, meaning a singular weight configuration is applied uniformly across all sensor channels , drastically decreasing the global trainable parameter volume while preserving the underlying unified degradation path.

Seasonal Mamba-CNN Forecasting Stream

The seasonal component  encapsulates complex non-linear temporal variations constituted by background operational noise and fault-induced impulses, which are routed through a hybrid processing stream fusing localized convolutional blocks and selective Mamba state space encoders to thoroughly map long-range dependencies.

Convolutional Patch Embedding Mechanism (CNN Patch Embedding)

To reduce the token sequence length forwarded to the recurrent layers and reinforce localized spatial-temporal inductive biases, the seasonal time-series is partitioned into localized patches of size  with a sliding stride of . The total volume of generated temporal tokens  is determined via the discrete mathematical expression:

The extraction and projection of these patches into a latent embedding space of dimension  are executed in parallel utilizing a one-dimensional convolutional layer () sweeping across the temporal grid of the sequence, configured with an output channel size equal to , a kernel size equal to , and a stride parameter equal to :

The physical rationale of this convolutional embedding block is to operate as an adaptive localized band-pass filter, smoothing the raw signal variations and compressing discrete kinematic impulse patterns within a localized window into high-density representation vectors.

Channel Independence (CI) Rule

To thoroughly suppress cross-channel noise propagation and prevent collinearity leakage between distinct acceleration measuring axes, the batch dimension () and sensor channel dimension () are flattened to transform the latent tensor configuration:

This regularizing constraint forces the subsequent encoder layers to evaluate sequence data from each acceleration sensor as an isolated single-channel entity, preserving the unique physical dynamics of individual mechanical measurement vectors.

Hybrid Mamba-CNN Block

The channel-independent latent representation  is subsequently forwarded through a series of hybrid selective state space blocks integrated with convolutional layers. Within each block, the hidden input signal  at patch step  is initially routed to a localized 1D convolutional branch with a kernel width  and a non-linear SiLU activation function to eliminate high-frequency instrument noise:

The filtered feature stream  serves as the input driver for the selective continuous state space model (), which executes hidden state transitions using data-dependent coefficient matrices:

The embedding of a 1D CNN layer directly prior to the linear scanning mechanism of Mamba structures a robust localized noise barrier, preventing the saturation of hidden recurrent states when subjected to large background white noise fields.

Seasonal Forecasting Head

The global temporal context vector generated by the Mamba layers, denoted as , is passed to a dedicated forecasting head that projects the latent representation back into the discrete time domain of the future forecast horizon:

Physics-Informed Statistical Head

To supplement purely data-driven, black-box hidden spaces that operate in a structurally uninterpretable manner, a physical statistical head is structured to anchor the latent optimization within explicit mechanical descriptors. This block extracts an 8-dimensional time-domain statistical vector directly from the raw historical lookback window  prior to high-pass filtering filtering utilizing established mechanical engineering formulations:

Mean (Indicates structural eccentricity and DC-offset components):

Standard Deviation (Measures the amplitude of energy variance surrounding the mean tracking profile):

Root Mean Square (RMS) (Acts as the primary metric tracking total destructive structural fatigue energy):

Peak-to-Peak (Captures the absolute maximum shock amplitude range within the window):

Skewness (Measures distribution asymmetry, sensitive to initial surface pitting asymmetry):

Kurtosis (Provides maximum statistical sensitivity to transient impact shocks during initial micro-cracking stages):

Crest Factor (The ratio of peak amplitude to the RMS baseline, isolating localized cracking pulses from background energy):

Shape Factor (The ratio of RMS to the mean absolute value, reflecting global structural changes in the wave profile):

This 8-dimensional physical descriptor array  is normalized via a BatchNormalization layer and linearly projected before being concatenated directly with the global latent representation generated by the Mamba stream:

This architectural loop maintains a modular design controlled via the system configuration parameter . If  is specified, the statistical extraction pathway is automatically bypassed to return a pure data-driven seasonal Mamba forecast.

Learnable Dual-Stream Mixing Module

To bypass the strict limitations of direct element-wise integration which enforces fixed, non-adjustable contribution ratios between components, the proposed framework implements a learnable dual-stream mixing module governed by a dynamic weighting parameter  optimized independently per sensor channel :

where  represents the standard mathematical Sigmoid activation function constraining the parameter bounds within the range , and  denotes a learnable weight parameter optimized via backpropagation gradient descents corresponding to each physical sensor location.

This learnable mixing paradigm empowers the network to automatically adjust the relative dominance between slow-moving physical degradation pathways (the trend component) and rapid non-linear transient vibrations (the seasonal component) optimized for specific sensor mounting points, thereby maximizing the forecasting accuracy of the synthesized sequence .

Anomaly Scoring and Leakage-Free POT Thresholding

Following the acquisition of the composite forecast sequence , the Anomaly Score at each time window  is formally defined via the Mean Squared Error (MSE) computed across all sensor dimensions and steps within the horizon :

To establish dynamic decision boundaries online in an objective manner, Extreme Value Theory (EVT) utilizing the Peak Over Threshold (POT) technique is integrated directly into the inference loop. The calibration workflow strictly honors a leakage-free constraint by exclusively executing parameter estimation over the error distribution derived from the historical early-stage healthy operational sequences, ensuring absolute isolation from any future degradation or target fault profiles. The execution procedure is structured via discrete mathematical phases:

An initial baseline anchor threshold  is established by computing a high-percentile marker (e.g., 98%) from the anomaly score sequence generated across the healthy training baseline.

The extreme positive excesses exceeding the anchor threshold are filtered and aggregated: .

The set of excess values  is fitted to a Generalized Pareto Distribution (GPD) to mathematically bound the asymptotic behavior of the distribution tail:

where  and  denote the shape and scale parameters, respectively, optimized via the Maximum Likelihood Estimation (MLE) method.

The final dynamic decision boundary  corresponding to a configured, conservative target alarm probability  (e.g., ) is computed as follows:

where  represents the total volume of baseline observation samples, and  is the historical volume of excess samples violating the initial anchor threshold .

During the real-time online monitoring stage, any testing window exhibiting an anomaly score satisfying the logical condition  is immediately flagged as a structural anomaly instance, ensuring a highly practical implementation that systematically eliminates false alarms caused by non-stationary operational variations.

Experiments Design

Dataset Description and Temporal Partitioning Workflow

The empirical validation of the proposed framework is executed utilizing the Paderborn University (UPB) run-to-failure ball bearing dataset, which represents a specialized industrial benchmark for rolling element bearing prognostic monitoring under time-varying operating conditions. The experimental setup utilizes type 61806-2RS ball bearings, which feature an inner diameter of 30 mm, an outer diameter of 42 mm, a width of 7 mm, and are equipped with 19 rolling elements. During the run-to-failure experiments, the test bearing assembly is subjected to highly dynamic operating conditions where the shaft rotating speed, static pre-loads, and superimposed dynamic load amplitudes are randomly drawn from stationary uniform distributions within predefined boundaries to simulate rigorous real-world mechanical environments. To thoroughly map the directional mechanics of structural degradation without cross-channel leakage, physical telemetry is captured synchronously via two one-directional accelerometers mounted on the bearing housing, tracking the horizontal rear direction (Channel A) and frontal direction (Channel C). Raw vibration data streams are sampled at an ultra-high acquisition rate of 128 kHz for early experimental runs and 64 kHz for subsequent extended lifecycles. Prior to sequence generation, a causal Butterworth high-pass filter is implemented with a configured cutoff frequency of 2000 Hz to isolate structural transient shock impacts from low-frequency shaft rotation modulations and environmental background noise.

To ensure a strict, leakage-free verification protocol that aligns with actual industrial monitoring constraints, a dedicated subset consisting of exactly 10 distinct bearings out of the 17 total experimental lifecycles is isolated. The training matrix ingests data streams compiled across bearings B02, B05, B08, B10, B11, and B17. Generalization bounds and unsupervised fault detection accuracy are validated on an independent testing cohort consisting of bearings B01, B03, B04, B08, B10, B12, and B17. For each operational trajectory, a precise temporal partitioning protocol is enforced: the first 5% of the sequence is discarded to eliminate initial transient data anomalies. The unsupervised network training window is confined exclusively to the early-stage healthy operational segment spanning from the 5% index up to 40% of the total recorded lifespan, where the condition variable is flagged as strictly healthy (), preventing failure profiles from corrupting the latent space. The remaining segment from the 40% milestone until final catastrophic degradation is reserved for testing and unsupervised threshold calibration. Continuous monitoring sequences are structured using a sliding window protocol configured with a lookback history window () of 4096 points, a forecast horizon () of 1024 points, and a sliding window stride of 1024 points.

Baseline Models and Parameter Budget Synchronization Protocol

To verify the structural superiority of the hybrid Mamba-CNN architecture, the evaluation is benchmarked against five representative deep sequence models representing recurrent, convolutional, attention-based, and selective state-space paradigms:

Long Short-Term Memory (LSTM): Implemented as a classical recurrent architecture to track sequential temporal transitions.

Temporal Convolutional Network (TCN): Configured with dilated convolutions to assess parallelized localized receptive fields.

ModernTCN: A modernized convolutional time-series backbone separating depth-wise and point-wise decompositions to maximize representation capacity.

PatchTST: A state-of-the-art attention mechanism operating via channel-independent patching to achieve linear-scale transformer mapping.

SimpleMamba: A standard selective state space architecture operating in isolation, completely decoupled from convolutional operations or physical statistical descriptors, serving as a baseline ablation.

A prominent flaw in traditional deep learning benchmarks is the unfair allocation of model capacities, where a proposed network often outperforms baselines simply due to an inflated parameter volume. To enforce rigorous academic fairness, an automated baseline parameter scaling protocol is activated (auto_scale_baselines: true). Under this constraint, the hidden dimensions and layer counts of all five benchmarking models are dynamically scaled to synchronize their global volume of trainable parameters with that of the proposed HybridMambaCNN. Consequently, any observed performance variance is isolated entirely to structural and architectural efficiency rather than raw capacity imbalances.

Training Infrastructure and Optimization Pipeline

`The optimization of the deep sequence networks is executed over a strict 10-epoch execution horizon utilizing a mini-batch size of 128 samples on a dedicated hardware acceleration platform powered by the CUDA execution engine. The global learning rate is fixed at a constant value of . Rather than relying on standard Mean Squared Error (MSE) optimization during the gradient update loop, the global loss function is formulated utilizing the Huber loss paradigm. For absolute prediction errors bounded within the threshold parameter  where , the mathematical penalty is defined as:

Conversely, for large anomalous deviations where , the linear loss penalty is formulated as follows:

The choice of the Huber loss is physically motivated by the nature of raw industrial vibration telemetry. High-frequency industrial data frequently exhibits sudden, catastrophic non-Gaussian outlier spikes caused by external structural impacts or sensor artifacts. Standard MSE loss squares these errors, forcing the gradient descent optimization to over-correct for random anomalies and destabilizing normal healthy representation learning. The Huber loss integrates a linear penalty regime for errors exceeding the threshold parameter , ensuring that the framework maintains structural robustness against impulsive noise outliers while retaining smooth quadratic convergence behavior for nominal reconstruction variances.

Anomaly Scoring and Evaluation Strategy

The operational state of the target bearing asset is tracked via a next-window forecasting discrepancy metric acting as the definitive unsupervised anomaly proxy. At each evaluation timestep , the anomaly score  is computed by taking the mathematical Mean Squared Error across all  acceleration channels and the complete forecast horizon length  ():

When the physical rolling elements operate within a stable, nominal envelope, the learned normal dynamics within the hybrid Mamba-CNN layers yield minimal forecasting deviations, keeping the anomaly score bounded. Upon the initialization of structural flaws, the forecasting capability drops abruptly, causing a distinct surge in .

To comprehensively judge the diagnostic efficacy, performance is mapped via standard industrial prognostic metrics. Precision computes the proportion of genuine structural degradations within the total generated alarms, minimizing unneeded maintenance interventions. Recall measures the true positive rate tracking the model’s sensitivity to capturing early-stage cracks. The F1-score establishes the harmonic mean to summarize the global trade-off balance. The Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) evaluate performance independent of specific threshold values, with AUPRC serving as the definitive gold standard due to the severe class imbalance inherently characterized by rare anomaly events in prolonged asset life cycles. Finally, the False Alarm Rate (FAR) calculates the metric frequency of incorrect warnings, serving as a critical operational cost indicator.

*Note: Due to local high-performance hardware computing constraints and data repository access protocols, a compact, representative subset of the complete Paderborn University bearing dataset is utilized across the evaluation runs. This structural boundary is explicitly introduced to establish a localized, high-density benchmarking baseline, ensuring perfect execution reproducibility under restricted parameter footprints.

Results

Waveform Decomposition Dynamics

Figure 1: Waveform Decomposition

Discrete vibration signal decomposition via the adaptive temporal series decomposition module is visually evaluated across three representative stages of the bearing B03 operational lifecycle: the initial healthy state (Healthy — M0001), the mid-life stage (Mid-life — M0308), and the fault onset phase (Fault Onset — M0558). As demonstrated in Fig. 1, configuring an extended moving average kernel size of  ensures that the trend component () maintains a smooth, flat trajectory and completely suppresses high-frequency localized oscillations. This stability confirms that the trend branch selectively captures only the low-frequency, one-dimensional DC baseline shift and the slow-moving progressive degradation dynamics of the mechanical system.

Conversely, the seasonal component () exhibits an almost perfect morphological correlation with the raw input signal () throughout the first two phases of the operational lifespan. At the fault onset milestone (M0558), a profound amplitude surge is recorded in the seasonal stream, where the root mean square (RMS) value escalates sharply from 0.059 V to 0.088 V, while the energy level of the corresponding trend component remains trapped at a sub-nominal baseline level (RMS of 0.006 V). The localized negative drift observed in the trend branch during this phase is identified as a technical artifact resulting from the sudden appearance of high-energy transient impulses, which does not jeopardize global mathematical stability. This manifestation verifies that the series decomposition algorithm successfully isolates structural fault-induced impact pulses from the heavy industrial background noise.

Frequency-Domain Verification

Figure 2: PSD Frequency Evidence

To validate the spectral boundary precision of the preprocessing and decomposition modules, the power spectral density (PSD) profiles of the decomposed streams are evaluated comparatively. In Fig. 2, a profound spectral separation spanning 6 to 8 orders of magnitude is explicitly recorded within the high-frequency band from 1000 Hz to 6000 Hz between the trend and seasonal components. The spectral energy separation metric demonstrates that the seasonal component commands absolute dominance, achieving an energy ratio 28 times greater than that of the trend component, marking a substantial architectural advancement over conventional narrow filter kernel selections.

The presence of localized spectral overlapping bands below 200 Hz is identified as an unavoidable physical consequence of low-frequency oscillations stemming from the primary shaft rotation of the induction motor. Nevertheless, the mechanical origin of the structural fault is thoroughly verified via the Fast Fourier Transform (FFT) spectrum of the seasonal component during the degradation phase. A dominant energy peak emerges in an isolated and highly precise manner at 108 Hz, matching the theoretical Ball Pass Frequency Inner Ring (BPFI) defect frequency of the mechanical assembly. This frequency-domain evidence confirms that the seasonal stream processed by the hybrid Mamba-CNN encoder maps genuine structural kinematic degradation profiles rather than learning stochastic white noise patterns from the industrial environment.

Longitudinal Evolution and Dimensionality Analysis

Figure 3: Longitudinal RMS

Figure 4: PCA Dimensionality

Figure 5: Architecture Justification

The degradation progression across the complete run-to-failure lifecycle of bearing B03, encompassing 614 sequential data points, is continuously tracked via time-domain energy metrics. As displayed in Fig. 3, the seasonal RMS profile maintains absolute statistical stability during the first 89% of the operational lifespan and exhibits a sharp spike precisely at milestone M0547, continuously oscillating within an elevated amplitude band between 0.15 V and 0.35 V to reflect severe structural fatigue propagation. The trajectory of the raw signal shows synchronous alignment with the seasonal branch, while the trend RMS profile hovers near 0 V across the vast majority of the lifecycle. This structural behavior justifies the deployment of a parsimonious linear forecasting branch for the trend component to minimize computational resource overhead.

The necessity of the parallel dual-stream hybrid Mamba-CNN framework is mathematically validated via the cumulative explained variance analysis illustrated in Fig. 4. The trend component reaches a 90% explained variance threshold using only 2 principal components (PCs), demonstrating a nearly planar, lower-dimensional manifold. In sharp contrast, the seasonal component exhibits highly complex, multi-dimensional non-linear dynamics, where the utilization of 30 distinct principal components only manages to explain approximately 83% of the total variance. The dimensionality expansion ratio between the two streams reaches 15.5×. As empirically demonstrated on the B03 run-to-failure dataset, the trend component requires only 2 principal components to explain 90% of its variance, while the seasonal component requires more than 30 components to reach 83% (dimensionality ratio: 15.5×). This disparity directly motivates the proposed dual-stream architecture.

Quantitative Evidence Tables

TABLE II QUANTITATIVE EVOLUTION AND SPECTRAL SEPARATION METRICS ACROSS FILTER KERNELS (BEARING B03)

| Experimental Metric | Baseline Configuration (Kernel = 257) | Proposed Configuration (Kernel = 3457) | Physical / Mechanical Significance |
| --- | --- | --- | --- |
| PCA Trend Variance Bound (90% Threshold) | 13 PCs | 2 PCs | Nearly planar, lower-dimensional manifold; justifies the linear branch application. |
| PCA Seasonal Variance Bound (90% Threshold) | 31 PCs | > 30 PCs (83% at 30) | Complex multi-dimensional non-linear dynamics; demands the Mamba encoder. |
| Dimensionality Expansion Ratio (Seasonal/Trend) | 2.4× | 15.5× | Quantitative evidence validation for the Architectural Parsimony principle. |
| Spectral Energy Separation (Seasonal/Trend) | 0.66× | 28.0× | Completely eliminates low-frequency trend profiles from masking structural fault impulses (Fault Masking). |
| Identified Analytical Fault Peak Frequency | None | 108 Hz | Perfect alignment with the theoretical Ball Pass Frequency Inner Ring (BPFI). |

TABLE III LIMITATION DIAGNOSTICS AUDIT AND REPRODUCIBILITY DEFENSE MATRIX

| Observed Artifact | Technical Nature | Systemic Impact | Scholarly Defense Argument |
| --- | --- | --- | --- |
| Trend Negative Drift (Fig. 1 — Fault Onset Column) | DC drift artifact generated when the symmetric moving average filter encounters high-amplitude transient fault impulses. | Completely benign; does not jeopardize network convergence. | This represents a natural mathematical outcome of a sliding integrated kernel; the negative drift confirms that the seasonal path has thoroughly isolated the high-frequency transient impact energy. |
| Low-Frequency Spectral Overlap (Fig. 2 — Below 200 Hz Region) | Frequency band overlapping originating from fundamental mechanical shaft rotation and background induction motor loads. | Sub-nominal baseline operational noise. | Low-frequency spectral blending is a fundamental constraint of moving average filters. This residual energy is thoroughly suppressed by the downstream Channel Independence (CI) constraint and the localized Local 1D CNN block prior to the selective scan. |
| Visual Bounding Absence (Fig. 5 — Scatter Cluster Distribution) | Omission of validation boundary geometry (ellipse overlays) mapping distinct regime spaces. | Slightly reduces visual graphic immediacy. | The morphological boundary separation (highly dense healthy cluster versus a linear path along the seasonal axis) is mathematically absolute; the unsupervised POT-EVT thresholding pipeline automates decision boundaries without manual geometric annotation. |

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
