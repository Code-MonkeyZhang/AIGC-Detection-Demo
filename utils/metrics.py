#!/usr/bin/env python3
"""
评估指标计算
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import List, Dict


def calculate_metrics(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Dict:
    """
    计算各项评估指标
    
    Args:
        y_true: 真实标签 (0 = Real, 1 = AI)
        y_pred: 预测标签 (0 = Real, 1 = AI)
        y_scores: 预测概率分数
        
    Returns:
        包含各项指标的字典
    """
    # 过滤掉失败的预测（None值）
    valid_indices = [i for i, (pred, score) in enumerate(zip(y_pred, y_scores)) 
                     if pred is not None and score is not None]
    
    if not valid_indices:
        return {
            "error": "所有预测都失败",
            "valid_samples": 0
        }
    
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    y_scores_valid = [y_scores[i] for i in valid_indices]
    
    # 计算指标
    metrics = {
        "valid_samples": len(valid_indices),
        "total_samples": len(y_true),
        "success_rate": len(valid_indices) / len(y_true),
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "precision": precision_score(y_true_valid, y_pred_valid, zero_division=0),
        "recall": recall_score(y_true_valid, y_pred_valid, zero_division=0),
        "f1_score": f1_score(y_true_valid, y_pred_valid, zero_division=0),
        "roc_auc": roc_auc_score(y_true_valid, y_scores_valid),
        "confusion_matrix": confusion_matrix(y_true_valid, y_pred_valid).tolist()
    }
    
    return metrics


def print_metrics(detector_name: str, metrics: Dict):
    """
    打印评估指标
    
    Args:
        detector_name: 检测器名称
        metrics: 指标字典
    """
    print(f"\n{'='*70}")
    print(f"📊 {detector_name} - 评估结果")
    print(f"{'='*70}")
    
    if "error" in metrics:
        print(f"❌ {metrics['error']}")
        return
    
    print(f"有效样本数: {metrics['valid_samples']}/{metrics['total_samples']} "
          f"({metrics['success_rate']:.1%})")
    print(f"\n分类指标:")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  精确率 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  召回率 (Recall):    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 分数:            {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    
    print(f"\n混淆矩阵:")
    cm = metrics['confusion_matrix']
    print(f"                预测 Real    预测 AI")
    print(f"  真实 Real     {cm[0][0]:>10}  {cm[0][1]:>10}")
    print(f"  真实 AI       {cm[1][0]:>10}  {cm[1][1]:>10}")
    
    print(f"\n解读:")
    if metrics['accuracy'] >= 0.9:
        print("  ✅ 准确率优秀 (≥90%)")
    elif metrics['accuracy'] >= 0.8:
        print("  ✓ 准确率良好 (80-90%)")
    else:
        print("  ⚠️  准确率需要改进 (<80%)")
    
    if metrics['f1_score'] >= 0.85:
        print("  ✅ F1分数优秀，模型平衡性好")
    elif metrics['f1_score'] >= 0.7:
        print("  ✓ F1分数良好")
    else:
        print("  ⚠️  F1分数较低，可能存在类别不平衡问题")

