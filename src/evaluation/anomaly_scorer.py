import torch
import torch.nn.functional as F

def calculate_anomaly_score(y_true, y_pred, metric='mse', normalized=False):
    """
    Tính Anomaly Score giữa chuỗi thực tế và chuỗi dự báo.
    y_true: Tensor (B, C, H)
    y_pred: Tensor (B, C, H)
    normalized: Nếu True, sử dụng Log-MSE để nén dải động của sai số.
    """
    if metric == 'mse':
        loss = F.mse_loss(y_pred, y_true, reduction='none')
    elif metric == 'mae':
        loss = F.l1_loss(y_pred, y_true, reduction='none')
    else:
        raise ValueError(f"Unsupported metric: {metric}")
        
    scores = loss.mean(dim=(1, 2))
    
    if normalized:
        scores = torch.log1p(scores)

    return scores.cpu().numpy()
