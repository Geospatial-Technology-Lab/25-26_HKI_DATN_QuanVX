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

# Min/Max values for normalization (configure these based on your data)
FEATURE_MIN_MAX = {
    'lulc': (0.0, 10.0),
    'Density_River': (0.0, 0.5),
    'Density_Road': (0.0, 0.8),
    'Distan2river_met': (0.0, 5000.0),
    'Distan2road_met': (0.0, 8000.0),
    'aspect': (0.0, 360.0),
    'curvature': (-5.0, 5.0),
    'dem': (0.0, 3000.0),
    'flowDir': (0.0, 255.0),
    'slope': (0.0, 90.0),
    'twi': (0.0, 25.0),
    'NDVI': (-1.0, 1.0),
    'rainfall': (0.0, 4000.0)
}

# Study area bounds (configure based on your study area)
# Format: (lon_min, lat_min, lon_max, lat_max)
STUDY_AREA_BOUNDS = (105.0, 20.0, 106.0, 21.0)  # Example for Vietnam region

# Validation: Ensure all mapping values are in features list
assert all(value in FEATURES for value in FEATURE_MAPPING.values()), \
    "Some mapped features are not in the standard features list"

# Validation: Ensure all features have min/max values
assert all(feature in FEATURE_MIN_MAX for feature in FEATURES), \
    "Some features are missing min/max values"

print("✅ Feature configuration loaded successfully")