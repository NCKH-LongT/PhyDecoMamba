import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from scipy.stats import genpareto

def calculate_threshold_pot(scores, q=1e-3, init_level=0.98):
    """
    Tính ngưỡng Peak Over Threshold (POT) dựa trên Extreme Value Theory.
    Bổ sung clip outlier để tránh nhiễu tập train làm vọt ngưỡng.
    """
    scores = np.array(scores)
    cap = np.percentile(scores, 99.9)
    scores = np.clip(scores, None, cap)
    
    t = np.quantile(scores, init_level)
    peaks = scores[scores > t] - t
    if len(peaks) == 0:
        return t
    try:
        c, loc, scale = genpareto.fit(peaks, floc=0)
        prob = q * len(scores) / len(peaks)
        if prob >= 1:
            return t
        z_q = t + genpareto.isf(prob, c, loc=0, scale=scale)
        return z_q
    except Exception:
        return t

def calculate_threshold_3sigma(healthy_scores):
    """
    Tính ngưỡng dự báo lỗi dựa trên nguyên tắc 3-sigma (Mean + 3*Std) của tập Healthy.
    """
    mu = np.mean(healthy_scores)
    sigma = np.std(healthy_scores)
    threshold = mu + 3 * sigma
    return threshold

def calculate_threshold_robust(healthy_scores, k=3):
    """
    Tính ngưỡng dựa trên Median và IQR (Chống nhiễu và outliers tốt hơn 3-sigma).
    """
    median = np.median(healthy_scores)
    q1 = np.percentile(healthy_scores, 25)
    q3 = np.percentile(healthy_scores, 75)
    iqr = q3 - q1
    threshold = median + k * iqr
    return threshold

def find_best_threshold(anomaly_scores, true_labels, num_steps=1000):
    """
    Tự động tìm ngưỡng tối ưu bằng cách quét qua nhiều giá trị trên tập dữ liệu có nhãn.
    """
    best_f1 = -1
    best_threshold = 0
    
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    thresholds = np.linspace(min_score, max_score, num_steps)
    
    for t in thresholds:
        preds = (anomaly_scores > t).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    return best_threshold, best_f1

def calculate_threshold_percentile(healthy_scores, q=99.7):
    """
    Tính ngưỡng dựa trên phân vị (Percentile).
    """
    return np.percentile(healthy_scores, q)

from sklearn.mixture import GaussianMixture

def calculate_threshold_gmm(anomaly_scores):
    """
    Tự học ngưỡng bằng Gaussian Mixture Model (GMM).
    Giả định dữ liệu gồm 2 cụm: Normal và Abnormal.
    """
    scores = np.array(anomaly_scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(scores)
    
    means = gmm.means_.flatten()
    normal_idx = np.argmin(means)
    abnormal_idx = 1 - normal_idx
    
    threshold = (means[normal_idx] + means[abnormal_idx]) / 2
    return threshold

def calculate_metrics(anomaly_scores, true_labels, threshold):
    """
    Tính toán các chỉ số: Detection Delay, FAR (False Alarm Rate), F1, AUC
    """
    preds = (anomaly_scores > threshold).astype(int)
    f1 = f1_score(true_labels, preds, zero_division=0)
    
    if len(np.unique(true_labels)) > 1:
        try:
            auc = roc_auc_score(true_labels, anomaly_scores)
        except ValueError:
            auc = 0.5
    else:
        auc = 0.0
        
    cm = confusion_matrix(true_labels, preds, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-8)
    else:
        far = 0.0
        
    fault_idx = np.argmax(true_labels == 1) if 1 in true_labels else -1
    pred_faults = np.where((preds == 1) & (true_labels == 1))[0]
    pred_fault_idx = pred_faults[0] if len(pred_faults) > 0 else -1
    
    if fault_idx != -1 and pred_fault_idx != -1:
        delay = max(0, int(pred_fault_idx - fault_idx))
    else:
        delay = -1
        
    return {
        "F1": f1,
        "AUC": auc,
        "FAR": far,
        "Delay": delay,
        "Threshold": threshold,
        "CM": cm.tolist()
    }
