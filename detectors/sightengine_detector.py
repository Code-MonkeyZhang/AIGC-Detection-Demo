#!/usr/bin/env python3
"""
SightEngine API 检测器
"""

import requests
from typing import Optional
from .base_detector import BaseDetector


class SightEngineDetector(BaseDetector):
    """基于 SightEngine API 的检测器"""
    
    def __init__(self, api_user: str, api_secret: str, timeout: int = 30):
        """
        初始化 SightEngine 检测器
        
        Args:
            api_user: API 用户 ID
            api_secret: API 密钥
            timeout: 请求超时时间（秒）
        """
        super().__init__("SightEngine")
        self.api_user = api_user
        self.api_secret = api_secret
        self.timeout = timeout
        self.api_url = "https://api.sightengine.com/1.0/check.json"
        self.load_model()
    
    def load_model(self):
        """验证 API 配置"""
        if not self.api_user or not self.api_secret:
            raise ValueError("SightEngine API credentials not provided")
        print(f"✅ {self.name} 检测器已初始化")
    
    def detect(self, image_path: str) -> Optional[float]:
        """
        使用 SightEngine API 检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            AI 生成概率 (0-1)
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
                    timeout=self.timeout
                )
                result = response.json()
                
                if result.get("status") == "success":
                    return result["type"]["ai_generated"]
                else:
                    return None
                    
        except Exception:
            return None

