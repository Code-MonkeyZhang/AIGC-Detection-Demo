#!/usr/bin/env python3
"""
检测器基类
定义统一的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseDetector(ABC):
    """AI 图片检测器基类"""
    
    def __init__(self, name: str):
        """
        初始化检测器
        
        Args:
            name: 检测器名称
        """
        self.name = name
    
    @abstractmethod
    def detect(self, image_path: str) -> Optional[float]:
        """
        检测图片是否为 AI 生成
        
        Args:
            image_path: 图片路径
            
        Returns:
            AI 生成概率 (0-1)，失败返回 None
            0.0 = 完全真实
            1.0 = 完全是AI生成
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """加载模型或初始化 API 连接"""
        pass
    
    def get_name(self) -> str:
        """获取检测器名称"""
        return self.name
    
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        预测图片类别
        
        Args:
            image_path: 图片路径
            threshold: 分类阈值
            
        Returns:
            包含预测结果的字典
        """
        score = self.detect(image_path)
        
        if score is None:
            return {
                "score": None,
                "prediction": None,
                "error": True
            }
        
        return {
            "score": score,
            "prediction": 1 if score >= threshold else 0,  # 1 = AI, 0 = Real
            "error": False
        }

