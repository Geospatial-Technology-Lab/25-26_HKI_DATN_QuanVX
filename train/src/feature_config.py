FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

# Mapping từ tên feature trong .gdb sang tên trong FEATURES
FEATURE_NAME_MAPPING = {
    # Các feature có thể bị viết hoa chữ cái đầu trong .gdb
    'Aspect': 'aspect',
    'Dem': 'dem', 
    'Slope': 'slope',
    # Các feature khác giữ nguyên
    'lulc': 'lulc',
    'Density_River': 'Density_River',
    'Density_Road': 'Density_Road',
    'Distan2river_met': 'Distan2river_met',
    'Distan2road_met': 'Distan2road_met',
    'aspect': 'aspect',
    'curvature': 'curvature',
    'dem': 'dem',
    'flowDir': 'flowDir',
    'slope': 'slope',
    'twi': 'twi',
    'NDVI': 'NDVI',
    'rainfall': 'rainfall'
}

FEATURE_MIN_MAX = {
    'lulc': (0.0, 12.0),
    'Density_River': (0.0, 0.000675744),
    'Density_Road': (0.0, 16.5452),
    'Distan2river_met': (0.0, 12407.1),
    'Distan2road_met': (0.0, 14716.4),
    'aspect': (-1.0, 360.0),
    'curvature': (-17.4153, 16.4418),
    'dem': (-21.0, 1756.0),
    'flowDir': (0.0, 255.0),
    'slope': (0.0, 68.5592),
    'twi': (-0.94, 21.0),
    'NDVI': (-0.186454, 0.599315),
    'rainfall': (196.525, 1292.31)
}

STUDY_AREA_BOUNDS = (107.452349, 12.999731, 109.371059, 14.703494)

def normalize_feature_names(gdb_features: list) -> list:
    """
    Chuyển đổi tên feature từ .gdb sang tên chuẩn trong FEATURES
    
    Args:
        gdb_features: Danh sách tên feature từ file .gdb
        
    Returns:
        list: Danh sách tên feature đã được chuẩn hóa
    """
    normalized_features = []
    for feature in gdb_features:
        if feature in FEATURE_NAME_MAPPING:
            normalized_features.append(FEATURE_NAME_MAPPING[feature])
        else:
            normalized_features.append(feature)
    return normalized_features

def get_feature_mapping_dict(gdb_features: list) -> dict:
    """
    Tạo dictionary mapping từ tên feature trong .gdb sang index trong FEATURES
    
    Args:
        gdb_features: Danh sách tên feature từ file .gdb
        
    Returns:
        dict: Dictionary mapping {gdb_feature_name: target_index_in_FEATURES}
    """
    mapping = {}
    for gdb_feature in gdb_features:
        # Chuẩn hóa tên feature
        normalized_name = FEATURE_NAME_MAPPING.get(gdb_feature, gdb_feature)
        # Tìm index trong FEATURES
        if normalized_name in FEATURES:
            target_index = FEATURES.index(normalized_name)
            mapping[gdb_feature] = target_index
    return mapping