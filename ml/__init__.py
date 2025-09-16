"""
ML Framework cho tối ưu hóa mô hình dự đoán lũ lụt

Cung cấp framework tích hợp cho:
- Data processing và preprocessing
- Model training với nhiều algorithms
- Optimization với PSO, RSO, PUMA
- Evaluation và metrics tracking
"""

__version__ = "2.0.0"
__author__ = "Your Team"

from .utils.config import load_config
from .utils.logging import setup_logging

# Thiết lập logging mặc định
setup_logging()