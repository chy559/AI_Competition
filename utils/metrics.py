import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def calculate_metrics(y_true, y_pred):
    """计算二分类评估指标"""
    y_true = y_true.cpu().numpy()
    y_pred_proba = y_pred.cpu().detach().numpy()
    y_pred_label = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred_label),
        "f1": f1_score(y_true, y_pred_label)
    }
    return metrics