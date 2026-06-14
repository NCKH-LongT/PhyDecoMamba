import torch
import torch.nn as nn
import math


class SeriesDecomposition(nn.Module):
    """
    EMA-based Series Decomposition — khớp chính xác với Eq.1–2 bài báo TSP.

    X_trend[k] = α · X[k] + (1−α) · X_trend[k−1]
    X_seasonal  = X − X_trend

    α được học tự động qua sigmoid khi learnable=True:
        α = σ(θ_α),  θ_α ∈ ℝ là tham số học được end-to-end.

    Ưu điểm so với AvgPool1d (Autoformer/DLinear):
    - Không có "padding bias" tại biên chuỗi
    - Không cần chọn kernel_size thủ công
    - α được tối ưu hóa cùng toàn bộ model (converges to α ≈ 0.03 như báo cáo trong bài báo)
    """
    def __init__(self, alpha: float = 0.1, learnable: bool = True):
        super().__init__()
        if learnable:
            # Dùng logit(alpha) để sigmoid-bounded trong (0, 1) khi tối ưu hóa
            init_logit = math.log(alpha / (1.0 - alpha))
            self._alpha = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('_alpha', torch.tensor(alpha).clamp(0.01, 0.99))
        self.learnable = learnable

    @property
    def alpha(self):
        if self.learnable:
            return torch.sigmoid(self._alpha)
        return self._alpha

    def forward(self, x):
        """
        x      : (Batch, Channels, Length)
        returns: seasonal (B, C, L), trend (B, C, L)

        Dùng list + torch.stack thay vì ghi in-place vào tensor
        để tránh RuntimeError: inplace operation breaks autograd graph.
        """
        B, C, L = x.shape
        alpha = self.alpha

        # Tích luỹ từng bước EMA vào list — không có in-place write nào
        trend_steps = [x[:, :, 0]]          # (B, C) tại t=0
        for t in range(1, L):
            # trend[t] = α·x[t] + (1−α)·trend[t−1] — pure functional, no inplace
            trend_t = alpha * x[:, :, t] + (1.0 - alpha) * trend_steps[-1]
            trend_steps.append(trend_t)

        # Stack list of (B, C) → (B, C, L)
        trend = torch.stack(trend_steps, dim=-1)
        seasonal = x - trend
        return seasonal, trend


class SimplePatchEmbedding(nn.Module):
    """
    Simple Linear Patching — khớp với Eq.4–5 bài báo TSP.
    Tạo M = floor((L - P) / S) + 1 patches kích thước P, stride S.
    Chiếu vào latent space D chiều bằng Conv1D (Eq.5).
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
