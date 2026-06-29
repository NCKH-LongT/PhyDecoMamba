# PhyDecoMamba: Physics-Aware Decomposed Mamba for EVT-Calibrated Bearing Detection

This repository contains the official implementation of the **PhyDecoMamba** (Physics-Aware Decomposed Mamba) forecasting model for bearing fault diagnosis and run-to-failure anomaly detection.

The framework leverages physical statistics (Stats Head) and a Mamba-CNN architecture to forecast high-frequency vibration signals and compute anomaly scores, preserving absolute amplitude trends to detect degradation.

---

## 🌟 Key Features

1. **PhyDecoMamba Architecture**: Combines temporal series decomposition, space-localized convolution patches, and linear-complexity State Space Models (Mamba) for long-sequence predictive forecasting.
2. **Physics-Informed Stats Head**: Incorporates 8 time-domain physical features (RMS, Kurtosis, Crest Factor, Shape Factor, etc.) directly into the fusion head to guide forecasting using domain knowledge.
3. **Advanced Hardware-Aware Profiling**: Benchmarks models on:
   - Peak VRAM Consumption (MB)
   - 4-Step Real-time Latency (Data Transfer, Inference, Anomaly Scoring, Decision)
   - Calibration Overhead (ms)
4. **Fair Parameter Budgeting**: Features an auto-scaling utility that adjusts baseline model sizes (LSTM, ModernTCN, PatchTST, SimpleMamba) to align with the proposed hybrid model's parameter budget.
5. **No-Leakage Thresholding**: Calibrates adaptive alarm thresholds (3-Sigma, Robust, POT, GMM) strictly on the early healthy phase of each individual bearing.

---

## 📂 Project Structure

```text
├── configs/                  # YAML configurations (e.g., default.yaml, snano.yaml)
├── src/
├── data/                     # (Will be downloaded automatically)
├── results/                  # Results, checkpoints, and visualization plots
├── requirements.txt          # Python dependencies
├── running_guide.md          # Guide for environment setup, training, and evaluation
├── configuration_guide.md    # Detail on tuning parameters for optimal results
├── source_code_description.md# Overview of source code modules and data flow
└── paper_preparation_checklist.md # Academic writing checklist for Q1 journal submissions
```


---

## 📖 Getting Started

To install, configure, and run the models, please refer to the corresponding documentation:

*   **Installation & Usage**: See [running_guide.md](running_guide.md)
*   **Hyperparameter Tuning**: See [configuration_guide.md](configuration_guide.md)
*   **Architecture & Code Modules**: See [source_code_description.md](source_code_description.md)
*   **Manuscript Writing Guidelines**: See [paper_preparation_checklist.md](paper_preparation_checklist.md)

---

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
