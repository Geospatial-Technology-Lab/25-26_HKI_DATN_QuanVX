FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

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

# Format: (lon_min, lat_min, lon_max, lat_max)
STUDY_AREA_BOUNDS = (105.0, 20.0, 106.0, 21.0)

# Total number of data rows (configure this to avoid loading entire file for counting)
TOTAL_ROWS = 224_000_000  # 224 million rows

# Validation: Ensure all features have min/max values
assert all(feature in FEATURE_MIN_MAX for feature in FEATURES), \
    "Some features are missing min/max values"

print("✅ Feature configuration loaded successfully")