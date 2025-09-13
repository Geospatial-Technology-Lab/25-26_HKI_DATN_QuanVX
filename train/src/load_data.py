import geopandas as gpd
import numpy as np
import joblib
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.feature_names = [
            'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 
            'Distan2road_met', 'aspect', 'curvature', 'dem', 'flowDir', 
            'slope', 'twi', 'NDVI', 'rainfall'
        ]
    
    def load_data(self, data_path):
        gdf = gpd.read_file(data_path)
        gdf = gdf[self.feature_names + ['geometry']].dropna()
        coordinates = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        features = gdf[self.feature_names].values
        # Chuẩn hóa Min-Max: (x - xmin) / (xmax - xmin)
        features_min = np.min(features, axis=0)
        features_max = np.max(features, axis=0)
        features = (features - features_min) / (features_max - features_min)
        return coordinates, features
    
    def load_all_models(self, model_dir):
        models = {}
        model_dir = Path(model_dir)
        model_files = list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl"))
        
        for model_file in model_files:
            model = joblib.load(model_file)
            models[model_file.stem] = model
                
        return models