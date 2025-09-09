"""
config.py - Cấu hình chung cho dự báo lũ lụt
"""

import os

# Đường dẫn
MODEL_DIR = r'D:\25-26_HKI_DATN_QuanVX\train\model'
DATA_FILE = 'data/flood_points.csv'
OUTPUT_DIR = 'output'

PIXEL_SIZE = 0.00009  # độ (10m tại Việt Nam)
CRS = 'EPSG:4326'    # WGS84 Geographic

# Features cho bài toán regression
FEATURE_COLUMNS = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

# Thông số xuất TIFF
TIFF_COMPRESS = 'lzw'
TIFF_DTYPE = 'float32'
NODATA_VALUE = -9999.0  # Dùng -9999 thay vì np.nan cho tương thích