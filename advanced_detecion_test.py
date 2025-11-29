#!/usr/bin/env python3
"""
高级 AI 图片检测脚本
结合 SightEngine API 和 Nonescape Mini 模型
使用加权平均方法进行综合判断

使用方法:
    python advanced_detecion_test.py

配置:
    - 修改 SIGHTENGINE_API_USER 和 SIGHTENGINE_API_SECRET 为你的 API 凭证
    - 修改 MODEL_PATH 为你的 Nonescape mini 模型路径
    - 修改 TEST_PICS_DIR 为你的测试图片目录
    - 调整 WEIGHT_SIGHTENGINE 和 WEIGHT_NONESCAPE 来改变模型权重
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import requests
import torch
from PIL import Image
import json

# 添加 nonescape 模块路径
sys.path.insert(0, str(Path(__file__).parent / "src" / "nonescape" / "python"))
from nonescape import NonescapeClassifierMini, preprocess_image

# ==================== 配置区域 ====================

# SightEngine API 配置
SIGHTENGINE_API_USER = "1130240739"
SIGHTENGINE_API_SECRET = "2EMcFiUxsHjn6FbJyn2ZeDgKkKJBZDzM"

# Nonescape 模型路径
MODEL_PATH = "model/nonescape-mini-v0.safetensors"

# 测试图片目录
TEST_PICS_DIR = "test_pics"

# 权重配置 (总和应该为 1.0)
WEIGHT_SIGHTENGINE = 0.5  # SightEngine API 权重
WEIGHT_NONESCAPE = 0.5     # Nonescape Mini 权重

# API 超时设置 (秒)
API_TIMEOUT = 30

# ==================================================


class AdvancedAIDetector:
    """高级 AI 图片检测器，结合多个模型"""
    
    def __init__(self, model_path: str, api_user: str, api_secret: str):
        """
        初始化检测器
        
        Args:
            model_path: Nonescape mini 模型路径
            api_user: SightEngine API 用户ID
            api_secret: SightEngine API 密钥
        """
        self.api_user = api_user
        self.api_secret = api_secret
        self.api_url = "https://api.sightengine.com/1.0/check.json"
        
        # 加载 Nonescape mini 模型
        print(f"正在加载 Nonescape Mini 模型: {model_path}")
        try:
            self.model = NonescapeClassifierMini.from_pretrained(model_path)
            self.model.eval()
            print("✅ Nonescape Mini 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def detect_with_sightengine(self, image_path: str) -> Optional[float]:
        """
        使用 SightEngine API 检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            AI 生成概率 (0-1)，失败返回 None
        """
        try:
            with open(image_path, "rb") as image_file:
                files = {"media": image_file}
                data = {
                    "models": "genai",
                    "api_user": self.api_user,
                    "api_secret": self.api_secret,
                }
                
                response = requests.post(
                    self.api_url, 
                    files=files, 
                    data=data, 
                    timeout=API_TIMEOUT
                )
                result = response.json()
                
                if result.get("status") == "success":
                    return result["type"]["ai_generated"]
                else:
                    print(f"   ⚠️  SightEngine API 错误: {result}")
                    return None
                    
        except requests.exceptions.Timeout:
            print("   ⚠️  SightEngine API 请求超时")
            return None
        except Exception as e:
            print(f"   ⚠️  SightEngine API 异常: {e}")
            return None
    
    def detect_with_nonescape(self, image_path: str) -> Optional[float]:
        """
        使用 Nonescape Mini 模型检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            AI 生成概率 (0-1)，失败返回 None
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = preprocess_image(image)
            
            with torch.no_grad():
                probs = self.model(tensor.unsqueeze(0))
                ai_prob = probs[0][1].item()
                
            return ai_prob
            
        except Exception as e:
            print(f"   ⚠️  Nonescape 检测异常: {e}")
            return None
    
    def detect_combined(
        self, 
        image_path: str,
        weight_sightengine: float = WEIGHT_SIGHTENGINE,
        weight_nonescape: float = WEIGHT_NONESCAPE
    ) -> Dict:
        """
        结合两个模型进行检测
        
        Args:
            image_path: 图片路径
            weight_sightengine: SightEngine 权重
            weight_nonescape: Nonescape 权重
            
        Returns:
            包含检测结果的字典
        """
        result = {
            "filename": Path(image_path).name,
            "sightengine_score": None,
            "nonescape_score": None,
            "combined_score": None,
            "prediction": None,
            "confidence": None,
            "error": None
        }
        
        # SightEngine 检测
        sightengine_score = self.detect_with_sightengine(image_path)
        result["sightengine_score"] = sightengine_score
        
        # Nonescape 检测
        nonescape_score = self.detect_with_nonescape(image_path)
        result["nonescape_score"] = nonescape_score
        
        # 计算加权平均
        if sightengine_score is not None and nonescape_score is not None:
            combined = (
                weight_sightengine * sightengine_score + 
                weight_nonescape * nonescape_score
            )
            result["combined_score"] = combined
            
            # 确定预测结果
            if combined >= 0.7:
                result["prediction"] = "AI生成 (高置信度)"
                result["confidence"] = "高"
            elif combined >= 0.5:
                result["prediction"] = "AI生成 (中置信度)"
                result["confidence"] = "中"
            elif combined >= 0.3:
                result["prediction"] = "真实图片 (中置信度)"
                result["confidence"] = "中"
            else:
                result["prediction"] = "真实图片 (高置信度)"
                result["confidence"] = "高"
                
        elif sightengine_score is not None:
            # 只有 SightEngine 结果
            result["combined_score"] = sightengine_score
            result["prediction"] = "AI生成" if sightengine_score >= 0.5 else "真实图片"
            result["confidence"] = "低 (仅SightEngine)"
            result["error"] = "Nonescape 检测失败"
            
        elif nonescape_score is not None:
            # 只有 Nonescape 结果
            result["combined_score"] = nonescape_score
            result["prediction"] = "AI生成" if nonescape_score >= 0.5 else "真实图片"
            result["confidence"] = "低 (仅Nonescape)"
            result["error"] = "SightEngine 检测失败"
            
        else:
            # 两个都失败
            result["error"] = "所有检测方法都失败"
            result["prediction"] = "检测失败"
            
        return result


def format_score(score: Optional[float]) -> str:
    """格式化分数显示"""
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def print_result(result: Dict, index: int, total: int):
    """打印单个图片的检测结果"""
    print(f"\n[{index}/{total}] {result['filename']}")
    print("-" * 70)
    print(f"  SightEngine 分数:  {format_score(result['sightengine_score'])}")
    print(f"  Nonescape 分数:    {format_score(result['nonescape_score'])}")
    print(f"  综合分数:          {format_score(result['combined_score'])}")
    print(f"  预测结果:          {result['prediction']}")
    print(f"  置信度:            {result['confidence']}")
    if result['error']:
        print(f"  ⚠️  警告:           {result['error']}")


def print_summary(results: list, total_time: float):
    """打印汇总统计"""
    total = len(results)
    ai_generated = sum(1 for r in results if r['prediction'] and "AI生成" in r['prediction'])
    authentic = sum(1 for r in results if r['prediction'] and "真实图片" in r['prediction'])
    failed = sum(1 for r in results if r['prediction'] == "检测失败")
    
    print("\n" + "=" * 70)
    print("检测完成 - 汇总统计")
    print("=" * 70)
    print(f"总图片数:       {total}")
    print(f"AI生成图片:     {ai_generated} ({ai_generated/total*100:.1f}%)")
    print(f"真实图片:       {authentic} ({authentic/total*100:.1f}%)")
    print(f"检测失败:       {failed}")
    print(f"总耗时:         {total_time:.2f} 秒")
    print(f"平均每张:       {total_time/total:.2f} 秒")
    print("=" * 70)
    
    # 显示权重配置
    print(f"\n权重配置:")
    print(f"  SightEngine: {WEIGHT_SIGHTENGINE:.1%}")
    print(f"  Nonescape:   {WEIGHT_NONESCAPE:.1%}")


def save_results(results: list, output_file: str = "advanced_detection_results.json"):
    """保存结果到 JSON 文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n⚠️  保存结果失败: {e}")


def main():
    print("=" * 70)
    print("高级 AI 图片检测工具")
    print("结合 SightEngine API + Nonescape Mini 模型")
    print("=" * 70)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 '{MODEL_PATH}'")
        sys.exit(1)
    
    # 检查测试目录
    test_dir = Path(TEST_PICS_DIR)
    if not test_dir.exists():
        print(f"❌ 错误: 测试目录不存在 '{TEST_PICS_DIR}'")
        sys.exit(1)
    
    # 获取所有图片
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    images = [
        img for img in test_dir.iterdir()
        if img.is_file() and img.suffix in image_extensions
    ]
    
    if not images:
        print(f"❌ 错误: 在 '{TEST_PICS_DIR}' 中没有找到任何图片")
        sys.exit(1)
    
    print(f"\n找到 {len(images)} 张图片")
    print(f"权重配置: SightEngine={WEIGHT_SIGHTENGINE:.1%}, Nonescape={WEIGHT_NONESCAPE:.1%}\n")
    
    # 初始化检测器
    try:
        detector = AdvancedAIDetector(MODEL_PATH, SIGHTENGINE_API_USER, SIGHTENGINE_API_SECRET)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    # 开始检测
    results = []
    start_time = time.time()
    
    for idx, img_path in enumerate(images, 1):
        print(f"\n{'='*70}")
        print(f"正在检测 [{idx}/{len(images)}]: {img_path.name}")
        print(f"{'='*70}")
        
        img_start = time.time()
        result = detector.detect_combined(str(img_path))
        img_time = time.time() - img_start
        
        result["processing_time"] = img_time
        results.append(result)
        
        print_result(result, idx, len(images))
        print(f"  处理时间: {img_time:.2f} 秒")
    
    total_time = time.time() - start_time
    
    # 打印汇总
    print_summary(results, total_time)
    
    # 保存结果
    save_results(results)
    
    print("\n✅ 所有测试完成!")


if __name__ == "__main__":
    main()

