import torch
import torch.nn as nn
from .layers import SeriesDecomposition, SimplePatchEmbedding
from .mamba_encoder import MambaEncoder
from .fusion_head import FusionForecastHead


class PhyDecoMamba(nn.Module):
    """
    PhyDecoMamba (Physics-Aware Decomposed Mamba for EVT-Calibrated Bearing Detection).
    Formerly known as CI-Mamba++ or HybridMambaCNN.

    Kiến trúc (Figure 1 bài báo):
        [Input] (B, C, L)
            │
        [Series Decomposition — EMA learnable]  ← Eq.1–2

           ┌───────────────┐
      [Seasonal branch]  [Trend branch]
    SimplePatch → Mamba   Linear(L → H)    ← Eq.3–10
          │
      FusionHead + Stats ← Eq.19
           └───────────────┘
               [α-mix]                      ← Eq.20–21
            │
        [Output] (B, C, H)

    Cấu hình bài báo (Section 4):
    - RevIN: TẮT (bài báo không đề cập)
    - MultiScale patching: TẮT (dùng SimplePatchEmbedding đơn tầng, Eq.4–5)
    - Highpass filter: TẮT (tín hiệu thô)
    - Stats Head: BẬT (8 đặc trưng vật lý, Eq.19)
    - Series Decomp: BẬT (EMA-based, Eq.1–2)
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        data_cfg  = config.get('data', {})

        # --- Hyper-parameters ---
        in_channels  = model_cfg.get('in_channels',  data_cfg.get('input_dim', 1))
        patch_size   = model_cfg.get('patch_size',   data_cfg.get('patch_size', 16))
        stride       = model_cfg.get('patch_stride', model_cfg.get('stride', data_cfg.get('stride', 8)))
        d_model      = model_cfg['mamba_d_model']
        forecast_len = model_cfg['forecast_len']
        lookback     = model_cfg.get('lookback', data_cfg.get('lookback', 4096))

        # EMA decomposition params (Eq.1–2 bài báo)
        decomp_alpha     = model_cfg.get('decomp_alpha', 0.1)
        decomp_learnable = model_cfg.get('decomp_learnable', True)

        # --- Ablation Study Toggles ---
        self.use_decomposition = model_cfg.get('use_decomposition', True)
        self.use_stats = model_cfg.get('use_stats', True)

        # ------------------------------------------------------------------
        # 1. Series Decomposition — EMA-based (Eq.1–2)
        # ------------------------------------------------------------------
        self.decomp = SeriesDecomposition(alpha=decomp_alpha, learnable=decomp_learnable)

        # ------------------------------------------------------------------
        # 2. Seasonal Branch — SimplePatchEmbedding + Mamba CI (Eq.4–10)
        # ------------------------------------------------------------------
        self.patching = SimplePatchEmbedding(
            patch_size=patch_size,
            stride=stride,
            embed_dim=d_model
        )

        # Mamba backbone (Channel-Independent: channels folded into batch dim)
        mamba_kwargs = model_cfg.get('mamba_kwargs', {}).copy()
        if 'mamba_d_state' in model_cfg: mamba_kwargs.setdefault('d_state', model_cfg['mamba_d_state'])
        if 'mamba_d_conv'  in model_cfg: mamba_kwargs.setdefault('d_conv',  model_cfg['mamba_d_conv'])
        if 'mamba_expand'  in model_cfg: mamba_kwargs.setdefault('expand',  model_cfg['mamba_expand'])

        self.mamba = MambaEncoder(
            d_model=d_model,
            n_layer=model_cfg['mamba_n_layer'],
            version=model_cfg.get('mamba_version', 1),
            bidirectional=model_cfg.get('bidirectional', False),
            **mamba_kwargs
        )

        # Forecasting head cho seasonal branch (Eq.10 + Eq.19)
        self.seasonal_head = FusionForecastHead(
            d_model=d_model,
            forecast_len=forecast_len,
            out_channels=1,
            use_stats=self.use_stats
        )

        # ------------------------------------------------------------------
        # 3. Trend Branch — DLinear-style Linear (Eq.3) với optional AvgPool
        # ------------------------------------------------------------------
        self.trend_downsample = model_cfg.get('trend_downsample', 1)
        if self.trend_downsample > 1:
            self.trend_pool = nn.AvgPool1d(kernel_size=self.trend_downsample)
            self.trend_head = nn.Linear(lookback // self.trend_downsample, forecast_len)
        else:
            self.trend_head = nn.Linear(lookback, forecast_len)

        # ------------------------------------------------------------------
        # 4. Learnable mixing — Eq.20–21: α_c·Ŷ_t + (1−α_c)·Ŷ_s
        # ------------------------------------------------------------------
        self.mix_alpha = nn.Parameter(torch.full((in_channels,), 0.5))

    def forward(self, x, stats=None):
        """
        x     : (Batch, Channels, Length)   — raw vibration signal
        stats : (Batch, Channels, 8)        — physical stats [optional]
        →       (Batch, Channels, forecast_len)
        """
        B, C, L = x.shape

        # --- Series Decomposition (Eq.1–2) ---
        if self.use_decomposition:
            seasonal, trend = self.decomp(x)              # both (B, C, L)

            # ── Trend Branch (Eq.3) ───────────────────────────────────────
            if self.trend_downsample > 1:
                trend_pooled = self.trend_pool(trend)      # (B, C, L // downsample)
                trend_out = self.trend_head(trend_pooled)  # (B, C, forecast_len)
            else:
                trend_out = self.trend_head(trend)         # (B, C, forecast_len)
        else:
            seasonal = x
            trend_out = None

        # ── Seasonal Branch (Eq.4–10) ─────────────────────────────────────
        # Patching: (B, C, L) → (B, C, N, d_model)
        s = self.patching(seasonal)                    # (B, C, N, D)
        _, _, N, D = s.shape

        # Channel-Independent: fold C into batch (Eq.6)
        s = s.reshape(B * C, N, D)                    # (B*C, N, D)
        s = self.mamba(s)                              # (B*C, N, D)

        # Stats cho CI mode (Eq.19)
        stats_ci = None
        if stats is not None and self.use_stats:
            stats_ci = stats.reshape(B * C, -1)       # (B*C, 8)

        s_out = self.seasonal_head(s, stats=stats_ci)  # (B*C, 1, forecast_len)
        s_out = s_out.reshape(B, C, -1)                # (B, C, forecast_len)

        # ── Learnable Mixing (Eq.20–21) ───────────────────────────────────
        if self.use_decomposition:
            alpha = torch.sigmoid(self.mix_alpha).view(1, C, 1)   # (1, C, 1)
            forecast = alpha * s_out + (1.0 - alpha) * trend_out  # (B, C, forecast_len)
        else:
            forecast = s_out

        return forecast


# Alias for backward compatibility with older versions and notebooks
HybridMambaCNN = PhyDecoMamba
