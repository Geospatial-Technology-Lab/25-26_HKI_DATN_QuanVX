FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

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

# Validation: Ensure all mapping values are in features list
assert all(value in FEATURES for value in FEATURE_MAPPING.values()), \
    "Some mapped features are not in the standard features list"

print("✅ Feature configuration loaded successfully")