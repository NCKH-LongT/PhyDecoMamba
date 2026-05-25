import torch
import torch.nn as nn
from .layers import SeriesDecomposition, SimplePatchEmbedding
from .mamba_encoder import MambaEncoder
from .fusion_head import FusionForecastHead


class HybridMambaCNN(nn.Module):
    """
    CI-Mamba++ (Channel-Independent Mamba with Decomposition).
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        data_cfg  = config.get('data', {})

        # --- Hyper-parameters ---
        in_channels  = model_cfg.get('in_channels',  data_cfg.get('input_dim', 1))
        patch_size   = model_cfg.get('patch_size',   data_cfg.get('patch_size', 64))
        stride       = model_cfg.get('stride',       data_cfg.get('stride', 32))
        d_model      = model_cfg['mamba_d_model']
        forecast_len = model_cfg['forecast_len']
        lookback     = model_cfg.get('lookback', data_cfg.get('lookback', 1024))
        decomp_kernel = model_cfg.get('decomp_kernel', 25)

        # --- Ablation Study Toggles ---
        self.use_decomposition = model_cfg.get('use_decomposition', True)
        self.use_stats = model_cfg.get('use_stats', True)

        # ------------------------------------------------------------------
        # 1. Series Decomposition
        # ------------------------------------------------------------------
        self.decomp = SeriesDecomposition(kernel_size=decomp_kernel)

        # ------------------------------------------------------------------
        # 2. Seasonal Branch — Simple Linear Patching + Mamba
        # ------------------------------------------------------------------
        self.patching = SimplePatchEmbedding(
            patch_size=patch_size,
            stride=stride,
            embed_dim=d_model
        )

        # Mamba backbone
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

        # Forecasting head for seasonal branch
        self.seasonal_head = FusionForecastHead(
            d_model=d_model,
            forecast_len=forecast_len,
            out_channels=1,
            use_stats=self.use_stats
        )

        # ------------------------------------------------------------------
        # 3. Trend Branch — simple Linear (DLinear-style) with Avg Pooling
        # ------------------------------------------------------------------
        self.trend_downsample = model_cfg.get('trend_downsample', 1)
        if self.trend_downsample > 1:
            self.trend_pool = nn.AvgPool1d(kernel_size=self.trend_downsample)
            self.trend_head = nn.Linear(lookback // self.trend_downsample, forecast_len)
        else:
            self.trend_head = nn.Linear(lookback, forecast_len)

        # ------------------------------------------------------------------
        # 4. Learnable mix weight (α·seasonal + (1-α)·trend per channel)
        # ------------------------------------------------------------------
        self.mix_alpha = nn.Parameter(torch.full((in_channels,), 0.5))

    def forward(self, x, stats=None):
        """
        x     : (Batch, Channels, Length)
        stats : (Batch, Channels, 8)
        →       (Batch, Channels, forecast_len)
        """
        B, C, L = x.shape

        # --- Series Decomposition ---
        if self.use_decomposition:
            seasonal, trend = self.decomp(x)              # both (B, C, L)
            
            # Trend Branch
            if self.trend_downsample > 1:
                trend_pooled = self.trend_pool(trend)      # (B, C, L // downsample)
                trend_out = self.trend_head(trend_pooled)  # (B, C, forecast_len)
            else:
                trend_out = self.trend_head(trend)         # (B, C, forecast_len)
        else:
            seasonal = x
            trend_out = None

        # Seasonal Branch
        s = self.patching(seasonal)                    # (B, C, N, D)
        _, _, N, D = s.shape

        # Channel-Independent
        s = s.reshape(B * C, N, D)
        s = self.mamba(s)

        stats_ci = None
        if stats is not None and self.use_stats:
            stats_ci = stats.reshape(B * C, -1)

        s_out = self.seasonal_head(s, stats=stats_ci)  # (B*C, 1, forecast_len)
        s_out = s_out.reshape(B, C, -1)                # (B, C, forecast_len)

        # Learnable Mixing
        if self.use_decomposition:
            alpha = torch.sigmoid(self.mix_alpha).view(1, C, 1)
            forecast = alpha * s_out + (1.0 - alpha) * trend_out
        else:
            forecast = s_out

        return forecast
