# =============================================================================
# CONFIGURATION FOR FLOOD PREDICTION DATA (200M+ Points Optimized)
# =============================================================================

# Danh sách features chuẩn theo thứ tự trong mô hình đã huấn luyện
FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

# Mapping tên features từ file GDB sang tên chuẩn
FEATURE_MAPPING = {
    # Tên trong GDB -> Tên chuẩn trong mô hình
    'lulc': 'lulc',
    'denriv': 'Density_River',           # density river
    'extract_denr1': 'Density_Road',      # density road (alternative name)
    'Distan2river_met': 'Distan2river_met',
    'Distan2road_met': 'Distan2road_met', 
    'aspect': 'aspect',
    'curvature': 'curvature',
    'dem': 'dem',
    'flowDir': 'flowDir',
    'slope': 'slope', 
    'twi': 'twi',
    'NDVI': 'NDVI',
    'rainfall': 'rainfall',
}

# =============================================================================
# BIG DATA PROCESSING CONFIGURATION (200M+ Points)
# =============================================================================

# Mặc định cấu hình cho dataset lớn (200 triệu điểm)
DEFAULT_BIG_DATA_CONFIG = {
    'expected_data_size': 200_000_000,  # 200M điểm
    'default_batch_size': 2_000_000,    # 2M điểm per batch
    'default_tile_size': 8000,          # 8k pixels per tile
    'memory_efficient': True,
    'gpu_accelerated': True
}

# Validation: Ensure all mapping values are in features list
assert all(value in FEATURES for value in FEATURE_MAPPING.values()), \
    "Some mapped features are not in the standard features list"

print("✅ Feature configuration loaded successfully")
print(f"📊 Configured for {DEFAULT_BIG_DATA_CONFIG['expected_data_size']:,} data points")