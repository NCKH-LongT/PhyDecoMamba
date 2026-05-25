import torch
import torch.nn as nn

class ModernTCNBlock(nn.Module):
    """
    ModernTCN Block inspired by ICLR 2024 paper.
    Separates temporal (depthwise) and channel (pointwise) modeling.
    """
    def __init__(self, dim, kernel_size=17, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size, 
            padding=kernel_size // 2, 
            groups=dim
        )
        self.norm1 = nn.LayerNorm(dim)
        
        self.pw_conv1 = nn.Linear(dim, dim * expansion_factor)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(dim * expansion_factor, dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        
        return res + x

class ModernTCNForecaster(nn.Module):
    """
    ModernTCN Forecaster for time series.
    """
    def __init__(self, input_dim=2, d_model=64, num_layers=3, kernel_size=17, horizon=512):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        
        self.embedding = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, kernel_size=kernel_size) 
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
            
        avg_pool = torch.mean(x, dim=-1)
        max_pool, _ = torch.max(x, dim=-1)
        out = avg_pool + max_pool
        
        out = self.fc(out)
        out = out.view(-1, self.input_dim, self.horizon)
        return out
