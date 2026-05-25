import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """
    Series decomposition block: tách Trend và Seasonal/Residual.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        padding = kernel_size // 2
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        """
        x      : (Batch, Channels, Length)
        returns: seasonal (B, C, L), trend (B, C, L)
        """
        trend = self.moving_avg(x)
        if trend.shape[-1] != x.shape[-1]:
            trend = nn.functional.interpolate(
                trend, size=x.shape[-1], mode='linear', align_corners=False
            )
        seasonal = x - trend
        return seasonal, trend


class SimplePatchEmbedding(nn.Module):
    """
    Simple Linear Patching (Standard for PatchTST/iTransformer).
    """
    def __init__(self, patch_size=64, stride=64, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        B, C, L = x.shape
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        B, C, N, P = patches.shape
        patches = patches.reshape(-1, P)
        x = self.projection(patches)
        x = x.reshape(B, C, N, -1)
        return x  # (B, C, N, d_model)
