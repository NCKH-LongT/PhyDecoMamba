import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.mamba import PhyDecoMamba, HybridMambaCNN, SimpleMamba
from src.models.baselines.lstm import LSTMForecaster
from src.models.baselines.modern_tcn import ModernTCNForecaster
from src.models.baselines.patch_models import PatchTST
from src.data import BearingDataset
from src.evaluation.anomaly_scorer import calculate_anomaly_score

def visualize_file(file_name, model_path, config_path, model_type, device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    stride = 32
    
    patch_size = config['model'].get('patch_size', 64)
    patch_stride = config['model'].get('patch_stride', 32)
    trend_downsample = config['model'].get('trend_downsample', 1)
    
    def initialize_model(m_type):
        if m_type == "Mamba1-Hybrid":
            return HybridMambaCNN({
                'model': {
                    'mamba_version': 1,
                    'mamba_d_model': config['model'].get('mamba_d_model', 64), 
                    'mamba_n_layer': config['model'].get('mamba_n_layer', 4),
                    'mamba_d_state': config['model'].get('mamba_d_state', 16), 
                    'mamba_d_conv': config['model'].get('mamba_d_conv', 4), 
                    'mamba_expand': config['model'].get('mamba_expand', 2),
                    'forecast_len': horizon, 
                    'patch_size': patch_size, 
                    'stride': patch_stride,
                    'trend_downsample': trend_downsample,
                    'in_channels': 2, 'lookback': lookback,
                    'decomp_kernel': config['model'].get('decomp_kernel', 25), 
                    'use_decomposition': config['model'].get('use_decomposition', True),
                    'use_stats': config['model'].get('use_stats', True),
                },
                'data': {'patch_size': patch_size, 'stride': patch_stride, 'lookback': lookback}
            })
        elif m_type == "SimpleMamba":
            return SimpleMamba({
                'model': {
                    'mamba_version': config['model'].get('mamba_version', 1),
                    'mamba_d_model': config['model'].get('mamba_d_model', 64),
                    'mamba_n_layer': config['model'].get('mamba_n_layer', 4),
                    'mamba_d_state': config['model'].get('mamba_d_state', 16),
                    'mamba_d_conv': config['model'].get('mamba_d_conv', 4),
                    'mamba_expand': config['model'].get('mamba_expand', 2),
                    'forecast_len': horizon,
                    'patch_size': patch_size,
                    'stride': patch_stride,
                    'in_channels': 2,
                    'lookback': lookback,
                },
                'data': {'patch_size': patch_size, 'stride': patch_stride, 'lookback': lookback}
            })
        elif m_type == "LSTM":
            return LSTMForecaster(input_dim=2, hidden_dim=122, num_layers=3, horizon=horizon)
        elif m_type == "ModernTCN":
            return ModernTCNForecaster(input_dim=2, d_model=144, num_layers=3, kernel_size=17, horizon=horizon)
        elif m_type == "PatchTST":
            return PatchTST(in_channels=2, lookback=lookback, patch_size=16, stride=8, d_model=128, nhead=16, num_layers=3, horizon=horizon)
        else:
            raise ValueError(f"Unsupported model type: {m_type}")

    model = initialize_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    data_path = os.path.join(config['data']['processed_dir'], file_name)
    signal = torch.load(data_path, map_location='cpu', weights_only=True)
    
    dataset_temp = BearingDataset(config['data']['processed_dir'], lookback, horizon, stride, split='test')
    healthy_baseline = dataset_temp.healthy_rms_baseline
    fault_threshold = healthy_baseline * 3.0
    file_rms = dataset_temp.file_rms[file_name]
    is_faulty_file = file_rms > fault_threshold
    
    print(f"File: {file_name}")
    print(f"File RMS: {file_rms:.4f} | Healthy Baseline: {healthy_baseline:.4f} | Fault Threshold: {fault_threshold:.4f}")
    print(f"Labeled as Faulty: {is_faulty_file}")

    n_samples = signal.shape[1]
    window_indices = range(0, n_samples - lookback - horizon, stride)
    
    errors = []
    rms_values = []
    timestamps = []
    
    with torch.no_grad():
        for start in window_indices:
            end_x = start + lookback
            end_y = end_x + horizon
            
            x = signal[:, start:end_x].unsqueeze(0).to(device)
            y = signal[:, end_x:end_y].unsqueeze(0).to(device)
            
            if isinstance(model, HybridMambaCNN):
                # Calculate stats
                eps = 1e-8
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True) + eps
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
                peak2peak = x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]
                skewness = torch.mean(((x - mean) / std) ** 3, dim=-1, keepdim=True)
                kurtosis = torch.mean(((x - mean) / std) ** 4, dim=-1, keepdim=True)
                crest_factor = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / (rms + eps)
                shape_factor = rms / (torch.mean(torch.abs(x), dim=-1, keepdim=True) + eps)
                stats = torch.cat([mean, std, rms, peak2peak, skewness, kurtosis, crest_factor, shape_factor], dim=-1)
                
                y_pred = model(x, stats)
            else:
                y_pred = model(x)
            
            score = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False).item()
            errors.append(score)
            
            win_rms = torch.sqrt(torch.mean(x**2)).item()
            rms_values.append(win_rms)
            timestamps.append(end_x)
            
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    axes[0].plot(signal[0].numpy(), color='gray', alpha=0.5, label='Vibration Ch0')
    axes[0].set_title(f"Raw Signal - {file_name} ({'FAULTY' if is_faulty_file else 'HEALTHY'})")
    axes[0].legend()
    
    axes[1].plot(timestamps, rms_values, color='blue', label='Window RMS')
    axes[1].axhline(y=fault_threshold, color='red', linestyle='--', label='Fault Threshold (3x Baseline)')
    axes[1].set_title("Window RMS")
    axes[1].legend()
    
    axes[2].plot(timestamps, errors, color='red', label='MSE Reconstruction Error')
    axes[2].set_title(f"{model_type} Reconstruction Error")
    axes[2].set_xlabel("Sample Index")
    axes[2].set_yscale('log')
    axes[2].legend()
    
    plt.tight_layout()
    save_path = f"results/visualizations/ad_check_{file_name.split('.')[0]}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data_B02_M1100.pt")
    parser.add_argument("--model_path", type=str, default="results/models/mamba1_hybrid_default_best.pth")
    parser.add_argument("--model_type", type=str, default="Mamba1-Hybrid")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_file(args.file, args.model_path, args.config, args.model_type, device)
