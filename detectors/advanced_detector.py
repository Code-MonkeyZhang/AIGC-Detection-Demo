#!/usr/bin/env python3
"""
高级组合检测器
结合多个检测器的结果
"""

from typing import List, Optional, Dict
from .base_detector import BaseDetector


class AdvancedDetector(BaseDetector):
    """组合多个检测器的高级检测器"""
    
    def __init__(self, detectors: List[BaseDetector], weights: List[float] = None):
        """
        初始化高级检测器
        
        Args:
            detectors: 检测器列表
            weights: 各检测器的权重（如果为 None，则平均分配）
        """
        super().__init__("Advanced-Combined")
        self.detectors = detectors
        
        if weights is None:
            # 平均分配权重
            self.weights = [1.0 / len(detectors)] * len(detectors)
        else:
            if len(weights) != len(detectors):
                raise ValueError("权重数量必须与检测器数量一致")
            if abs(sum(weights) - 1.0) > 0.01:
                raise ValueError("权重之和必须为 1.0")
            self.weights = weights
        
        self.load_model()
    
    def load_model(self):
        """验证所有检测器已加载"""
        detector_names = [d.get_name() for d in self.detectors]
        print(f"✅ {self.name} 检测器已初始化")
        print(f"   组合检测器: {', '.join(detector_names)}")
        print(f"   权重配置: {dict(zip(detector_names, self.weights))}")
    
    def detect(self, image_path: str) -> Optional[float]:
        """
        使用组合方法检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            加权平均的 AI 生成概率 (0-1)
        """
        scores = []
        valid_weights = []
        
        for detector, weight in zip(self.detectors, self.weights):
            score = detector.detect(image_path)
            if score is not None:
                scores.append(score)
                valid_weights.append(weight)
        
        if not scores:
            return None
        
        # 重新归一化权重
        total_weight = sum(valid_weights)
        normalized_weights = [w / total_weight for w in valid_weights]
        
        # 计算加权平均
        combined_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
        return combined_score
    
    def detect_with_details(self, image_path: str) -> Dict:
        """
        获取详细的检测结果
        
        Args:
            image_path: 图片路径
            
        Returns:
            包含各检测器结果的详细信息
        """
        results = {
            "combined_score": None,
            "detector_scores": {}
        }
        
        for detector in self.detectors:
            score = detector.detect(image_path)
            results["detector_scores"][detector.get_name()] = score
        
        results["combined_score"] = self.detect(image_path)
        
        return results

