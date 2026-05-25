import torch
import torch.nn as nn

class FusionForecastHead(nn.Module):
    def __init__(self, d_model=128, forecast_len=64, out_channels=1, use_stats=True):
        """
        Giai đoạn 3: Dung hợp và trực tiếp xuất ra dự báo tương lai.
        """
        super().__init__()
        self.out_channels = out_channels
        self.forecast_len = forecast_len
        self.use_stats = use_stats
        
        if self.use_stats:
            self.stats_norm = nn.BatchNorm1d(8)
            self.projection = nn.Linear(d_model + 8, forecast_len * out_channels)
        else:
            self.projection = nn.Linear(d_model, forecast_len * out_channels)

    def forward(self, x, stats=None):
        # x: (Batch, Num_Patches, d_model)
        x_trans = x.transpose(1, 2)
        avg_pool = torch.mean(x_trans, dim=-1)
        max_pool, _ = torch.max(x_trans, dim=-1)
        
        combined = avg_pool + max_pool
        
        if self.use_stats and stats is not None:
            norm_stats = self.stats_norm(stats)
            combined = torch.cat([combined, norm_stats], dim=-1)
            
        forecast = self.projection(combined)
        
        if self.out_channels > 1:
            forecast = forecast.view(-1, self.out_channels, self.forecast_len)
        
        return forecast
