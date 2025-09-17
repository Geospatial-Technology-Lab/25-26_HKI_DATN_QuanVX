FEATURES = [
    'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
    'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
    'slope', 'twi', 'NDVI', 'rainfall'
]

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
TOTAL_ROWS = 224_000_000