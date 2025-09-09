"""
GPU Configuration - Tối giản
Cấu hình GPU đơn giản cho xử lý dữ liệu lũ lụt
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Kiểm tra GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class SimpleGPU:
    """GPU configuration đơn giản"""
    
    def __init__(self):
        self.available = GPU_AVAILABLE
        if self.available:
            try:
                self.device = cp.cuda.Device()
                print(f"GPU available: {self.device}")
            except:
                self.available = False
                print("GPU setup failed, using CPU")
        else:
            print("Using CPU processing")
    
    def to_gpu(self, array):
        """Chuyển array lên GPU"""
        if self.available:
            try:
                return cp.asarray(array)
            except:
                pass
        return array
    
    def to_cpu(self, array):
        """Chuyển array về CPU"""
        if self.available and hasattr(array, '__array_interface__'):
            try:
                return cp.asnumpy(array)
            except:
                pass
        return array
    
    def clear(self):
        """Xóa GPU memory"""
        if self.available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

# Global instance
gpu = SimpleGPU()
