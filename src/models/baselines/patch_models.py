import torch
import torch.nn as nn
from src.models.baselines.transformer_small import PositionalEncoding

class PatchTST(nn.Module):
    """
    PatchTST (Patch Time Series Transformer) - Channel-Independent Patch-based Transformer.
    """
    def __init__(self, in_channels=2, lookback=1024, patch_size=64, stride=64, 
                 d_model=64, nhead=4, num_layers=3, dropout=0.1, horizon=512, head_type='pooling'):
        super().__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.lookback = lookback
        self.patch_size = patch_size
        self.stride = stride
        self.head_type = head_type
        
        self.num_patches = (lookback - patch_size) // stride + 1
        self.embedding = nn.Linear(patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if self.head_type == 'pooling':
            self.head = nn.Linear(d_model, horizon)
        else:
            self.head = nn.Linear(d_model * self.num_patches, horizon)
        
    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B * C, L)
        
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        x = self.embedding(x) # (B * C, num_patches, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        if self.head_type == 'pooling':
            x = x.mean(dim=1)
        else:
            x = x.reshape(B * C, -1)
        x = self.head(x)
        
        x = x.reshape(B, C, self.horizon)
        return x

VanillaTransformer = PatchTST
