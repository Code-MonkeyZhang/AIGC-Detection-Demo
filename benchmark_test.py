#!/usr/bin/env python3
"""
AI 图片检测器基准测试脚本

在 wildData 数据集上评估不同检测器的性能

使用方法:
    python benchmark_test.py [--mode MODE] [--samples N]
    
    --mode: all|sightengine|mini|full|advanced (默认: all)
    --samples: 测试样本数 (默认: 100)
"""

import os
import sys
import random
import pandas as pd
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 导入检测器
from detectors import (
    SightEngineDetector,
    NonescapeMiniDetector,
    NonescapeFullDetector,
    AdvancedDetector
)
from utils.metrics import calculate_metrics, print_metrics


# ==================== 配置区域 ====================

# SightEngine API 配置
SIGHTENGINE_API_USER = "1130240739"
SIGHTENGINE_API_SECRET = "2EMcFiUxsHjn6FbJyn2ZeDgKkKJBZDzM"

# 模型路径
MODEL_MINI_PATH = "model/nonescape-mini-v0.safetensors"
MODEL_FULL_PATH = "model/nonescape-v0.safetensors"

# 数据集配置
DATASET_DIR = "data/wildData/full_dataset"
TRAIN_CSV = "data/wildData/full_dataset/train.csv"

# 结果保存目录
RESULTS_DIR = "benchmark_results"

# 随机种子
RANDOM_SEED = 42

# 组合检测器权重配置
ADVANCED_WEIGHTS = {
    "SightEngine": 0.5,
    "Nonescape-Mini": 0.5
}

# ==================================================


def load_dataset(csv_path: str, dataset_dir: str, num_samples: int, seed: int) -> pd.DataFrame:
    """
    加载数据集并随机采样
    
    Args:
        csv_path: CSV 文件路径
        dataset_dir: 数据集根目录
        num_samples: 采样数量
        seed: 随机种子
        
    Returns:
        采样后的 DataFrame
    """
    print(f"\n📂 加载数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   总样本数: {len(df)}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 随机采样
    if num_samples < len(df):
        sample_df = df.sample(n=num_samples, random_state=seed)
        print(f"   随机采样: {num_samples} 个样本")
    else:
        sample_df = df
        print(f"   使用全部样本")
    
    # 添加完整路径
    sample_df['full_path'] = sample_df['file_name'].apply(
        lambda x: os.path.join(dataset_dir, x)
    )
    
    # 统计标签分布
    label_counts = sample_df['label'].value_counts()
    print(f"   标签分布:")
    print(f"     Real (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(sample_df)*100:.1f}%)")
    print(f"     AI (1):   {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(sample_df)*100:.1f}%)")
    
    return sample_df


def test_detector(detector, sample_df: pd.DataFrame, desc: str) -> Dict:
    """
    测试单个检测器
    
    Args:
        detector: 检测器实例
        sample_df: 样本数据
        desc: 进度条描述
        
    Returns:
        测试结果字典
    """
    print(f"\n🔍 开始测试: {detector.get_name()}")
    
    y_true = []
    y_pred = []
    y_scores = []
    failed_count = 0
    
    start_time = time.time()
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=desc):
        img_path = row['full_path']
        true_label = row['label']
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"   ⚠️  文件不存在: {img_path}")
            y_true.append(true_label)
            y_pred.append(None)
            y_scores.append(None)
            failed_count += 1
            continue
        
        # 进行预测
        result = detector.predict(img_path)
        
        y_true.append(true_label)
        y_pred.append(result['prediction'])
        y_scores.append(result['score'])
        
        if result['error']:
            failed_count += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"   完成时间: {elapsed_time:.2f} 秒")
    print(f"   平均每张: {elapsed_time/len(sample_df):.2f} 秒")
    if failed_count > 0:
        print(f"   ⚠️  失败数量: {failed_count}")
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    metrics['elapsed_time'] = elapsed_time
    metrics['avg_time_per_image'] = elapsed_time / len(sample_df)
    
    return {
        'detector_name': detector.get_name(),
        'metrics': metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        }
    }


def save_results(results: List[Dict], mode: str):
    """保存测试结果"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    simplified_results = []
    for result in results:
        simplified = {
            'detector_name': result['detector_name'],
            'metrics': result['metrics']
        }
        simplified_results.append(simplified)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"{mode}_{timestamp}.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n⚠️  保存结果失败: {e}")


def print_comparison(results: List[Dict]):
    """
    打印对比结果
    
    Args:
        results: 所有测试结果
    """
    print(f"\n{'='*70}")
    print("📈 性能对比总结")
    print(f"{'='*70}")
    
    print(f"\n{'检测器':<20} {'准确率':<12} {'F1分数':<12} {'ROC-AUC':<12} {'平均耗时'}")
    print("-" * 70)
    
    for result in results:
        name = result['detector_name']
        metrics = result['metrics']
        
        if 'error' not in metrics:
            print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
                  f"{metrics['roc_auc']:<12.4f} {metrics['avg_time_per_image']:.2f}s")
    
    # 找出最佳模型
    print(f"\n{'='*70}")
    print("🏆 最佳模型")
    print(f"{'='*70}")
    
    valid_results = [r for r in results if 'error' not in r['metrics']]
    
    if valid_results:
        best_accuracy = max(valid_results, key=lambda x: x['metrics']['accuracy'])
        best_f1 = max(valid_results, key=lambda x: x['metrics']['f1_score'])
        best_auc = max(valid_results, key=lambda x: x['metrics']['roc_auc'])
        fastest = min(valid_results, key=lambda x: x['metrics']['avg_time_per_image'])
        
        print(f"  最高准确率: {best_accuracy['detector_name']} "
              f"({best_accuracy['metrics']['accuracy']:.4f})")
        print(f"  最高F1分数: {best_f1['detector_name']} "
              f"({best_f1['metrics']['f1_score']:.4f})")
        print(f"  最高ROC-AUC: {best_auc['detector_name']} "
              f"({best_auc['metrics']['roc_auc']:.4f})")
        print(f"  最快速度: {fastest['detector_name']} "
              f"({fastest['metrics']['avg_time_per_image']:.2f}s/图)")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="AI 图片检测器基准测试")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'sightengine', 'mini', 'full', 'advanced'],
                        help='测试模式: all|sightengine|mini|full|advanced')
    parser.add_argument('--samples', type=int, default=100,
                        help='测试样本数')
    args = parser.parse_args()
    
    print("=" * 70)
    print("AI 图片检测器基准测试")
    print(f"测试模式: {args.mode}")
    print(f"样本数: {args.samples}")
    print("=" * 70)
    
    # 1. 加载数据集
    sample_df = load_dataset(TRAIN_CSV, DATASET_DIR, args.samples, RANDOM_SEED)
    
    # 2. 初始化检测器
    print(f"\n{'='*70}")
    print("🔧 初始化检测器")
    print(f"{'='*70}")
    
    mode = args.mode
    detectors = []
    detector_configs = []
    
    # SightEngine
    if mode in ['all', 'sightengine', 'advanced']:
        try:
            sightengine = SightEngineDetector(SIGHTENGINE_API_USER, SIGHTENGINE_API_SECRET)
            detectors.append(sightengine)
            if mode in ['all', 'sightengine']:
                detector_configs.append(('sightengine', sightengine))
        except Exception as e:
            print(f"⚠️  SightEngine 初始化失败: {e}")
    
    # Nonescape Mini
    if mode in ['all', 'mini', 'advanced']:
        try:
            if os.path.exists(MODEL_MINI_PATH):
                nonescape_mini = NonescapeMiniDetector(MODEL_MINI_PATH)
                detectors.append(nonescape_mini)
                if mode in ['all', 'mini']:
                    detector_configs.append(('mini', nonescape_mini))
            else:
                print(f"⚠️  找不到 Mini 模型: {MODEL_MINI_PATH}")
        except Exception as e:
            print(f"⚠️  Nonescape Mini 初始化失败: {e}")
    
    # Nonescape Full
    if mode in ['all', 'full']:
        try:
            if os.path.exists(MODEL_FULL_PATH):
                nonescape_full = NonescapeFullDetector(MODEL_FULL_PATH)
                detectors.append(nonescape_full)
                detector_configs.append(('full', nonescape_full))
            else:
                print(f"⚠️  找不到 Full 模型: {MODEL_FULL_PATH}")
        except Exception as e:
            print(f"⚠️  Nonescape Full 初始化失败: {e}")
    
    # Advanced (组合)
    if mode in ['all', 'advanced']:
        try:
            sightengine_det = None
            mini_det = None
            
            for d in detectors:
                if d.get_name() == "SightEngine":
                    sightengine_det = d
                elif d.get_name() == "Nonescape-Mini":
                    mini_det = d
            
            if sightengine_det and mini_det:
                advanced_weights = [
                    ADVANCED_WEIGHTS.get("SightEngine", 0.5),
                    ADVANCED_WEIGHTS.get("Nonescape-Mini", 0.5)
                ]
                total = sum(advanced_weights)
                advanced_weights = [w/total for w in advanced_weights]
                
                advanced = AdvancedDetector([sightengine_det, mini_det], advanced_weights)
                detector_configs.append(('advanced', advanced))
            else:
                print(f"⚠️  Advanced 需要 SightEngine 和 Nonescape-Mini")
        except Exception as e:
            print(f"⚠️  Advanced 初始化失败: {e}")
    
    if not detector_configs:
        print("❌ 没有可用的检测器！")
        sys.exit(1)
    
    # 3. 运行测试
    results = []
    for name, detector in detector_configs:
        result = test_detector(detector, sample_df, f"测试 {detector.get_name()}")
        results.append(result)
        print_metrics(detector.get_name(), result['metrics'])
    
    # 4. 打印对比结果
    print_comparison(results)
    
    # 5. 保存结果
    save_results(results, args.mode)
    
    print(f"\n{'='*70}")
    print("✅ 所有测试完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

