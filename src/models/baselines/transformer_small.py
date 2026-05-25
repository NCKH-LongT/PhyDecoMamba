import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        d_model = x.size(2)
        return x + self.pe[:, :seq_len, :d_model]

class iTransformer(nn.Module):
    """
    iTransformer (Inverted Transformer) - ICLR 2024.
    """
    def __init__(self, input_dim=2, lookback=1024, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, horizon=512):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.lookback = lookback
        
        self.embedding = nn.Linear(lookback, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, horizon)
        
    def forward(self, x):
        x = self.embedding(x) # (batch_size, input_dim, d_model)
        out = self.transformer(x) # (batch_size, input_dim, d_model)
        pred = self.head(out) # (batch_size, input_dim, horizon)
        return pred
