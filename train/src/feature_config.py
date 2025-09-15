FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

# Min/Max values for normalization (configure these based on your data)
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
    'twi': (-2.0, 21.0),
    'NDVI': (-0.186454, 0.599315),
    'rainfall': (196.525, 1292.31)
}

# Format: (lon_min, lat_min, lon_max, lat_max)
STUDY_AREA_BOUNDS = (105.0, 20.0, 106.0, 21.0)

# Total number of data rows (configure this to avoid loading entire file for counting)
TOTAL_ROWS = 224_000_000  # 224 million rows

# Validation: Ensure all features have min/max values
assert all(feature in FEATURE_MIN_MAX for feature in FEATURES), \
    "Some features are missing min/max values"

print("✅ Feature configuration loaded successfully")