import geopandas as gpd
import pandas as pd
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
        file_ext = Path(data_path).suffix.lower()
        
        if file_ext == '.gpd':
            return self.load_gpd_data(data_path)
        elif file_ext == '.csv':
            return self.load_csv_data(data_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return None, None
    
    def load_gpd_data(self, data_path):
        gdf = gpd.read_file(data_path)
        coordinates = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        features = gdf[self.feature_names].values
        return coordinates, features
    
    def load_csv_data(self, data_path):
        df = pd.read_csv(data_path)
        coordinates = df[['lat', 'lon']].values
        features = df[self.feature_names].values
        return coordinates, features
    
    def load_all_models(self, model_dir):
        models = {}
        for model_file in Path(model_dir).glob("*.joblib"):
            try:
                model = joblib.load(model_file)
                models[model_file.stem] = model
            except Exception as e:
                print(f"Error loading {model_file.name}: {e}")
        return models