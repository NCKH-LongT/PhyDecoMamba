import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, horizon=512, dropout=0.2):
        """
        Mô hình Baseline LSTM cho bài toán Dự báo Chuỗi thời gian (Forecasting).
        """
        super(LSTMForecaster, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, horizon * input_dim)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        # x: (batch_size, input_dim, lookback) -> transpose to (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] # (batch_size, hidden_dim)
        
        last_out = self.ln(last_out)
        last_out = self.dropout(last_out)
        
        pred = self.fc(last_out) # (batch_size, horizon * input_dim)
        pred = pred.view(-1, self.input_dim, self.horizon)
        return pred
