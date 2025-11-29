#!/usr/bin/env python3
"""
Nonescape Full 模型检测器
"""

import sys
from pathlib import Path
import torch
from PIL import Image
from typing import Optional

# 添加 nonescape 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "nonescape" / "python"))
from nonescape import NonescapeClassifier, preprocess_image

from .base_detector import BaseDetector


class NonescapeFullDetector(BaseDetector):
    """基于 Nonescape Full 模型的检测器"""
    
    def __init__(self, model_path: str):
        """
        初始化 Nonescape Full 检测器
        
        Args:
            model_path: 模型文件路径
        """
        super().__init__("Nonescape-Full")
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载 Nonescape Full 模型"""
        try:
            print(f"正在加载 {self.name} 模型: {self.model_path}")
            self.model = NonescapeClassifier.from_pretrained(self.model_path)
            self.model.eval()
            print(f"✅ {self.name} 模型加载成功")
        except Exception as e:
            print(f"❌ {self.name} 模型加载失败: {e}")
            raise
    
    def detect(self, image_path: str) -> Optional[float]:
        """
        使用 Nonescape Full 模型检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            AI 生成概率 (0-1)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = preprocess_image(image)
            
            with torch.no_grad():
                probs = self.model(tensor.unsqueeze(0))
                ai_prob = probs[0][1].item()
                
            return ai_prob
            
        except Exception:
            return None

