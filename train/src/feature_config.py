FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

FEATURE_MAPPING = {
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
    'rainfall': 'rainfall',
}

# Validation: Ensure all mapping values are in features list
assert all(value in FEATURES for value in FEATURE_MAPPING.values()), \
    "Some mapped features are not in the standard features list"

print("✅ Feature configuration loaded successfully")