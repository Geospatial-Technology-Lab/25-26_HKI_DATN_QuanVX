"""
Flood Prediction Program
Coordinate System: WGS1984 (EPSG:4326)
Pixel size: 0.00009
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI
from load_data import DataLoader

class FloodPredictor:
    def __init__(self):
        self.pixel_size = 0.00009
        self.crs = CRS.from_epsg(4326)
        self.data_loader = DataLoader()
        
    def load_data(self, data_file):
        return self.data_loader.load_data(data_file)
    
    def create_tif_grid(self, coordinates, predictions, output_path, model_name):
        try:
            min_x, max_x = coordinates[:, 0].min(), coordinates[:, 0].max()
            min_y, max_y = coordinates[:, 1].min(), coordinates[:, 1].max()
            
            width = int((max_x - min_x) / self.pixel_size) + 1
            height = int((max_y - min_y) / self.pixel_size) + 1
            
            grid = np.full((height, width), np.nan, dtype=np.float32)
            
            for i, (x, y) in enumerate(coordinates):
                col = int((x - min_x) / self.pixel_size)
                row = int((max_y - y) / self.pixel_size)
                if 0 <= row < height and 0 <= col < width:
                    grid[row, col] = predictions[i]
            
            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width,
                             count=1, dtype=np.float32, crs=self.crs, transform=transform, nodata=np.nan) as dst:
                dst.write(grid, 1)
            
            print(f"Created TIFF: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error creating TIFF for {model_name}: {e}")
            return None
    
    def run_prediction_pipeline(self, data_file, model_dir, results_dir):
        print("FLOOD PREDICTION PIPELINE")
        
        coordinates, features = self.load_data(data_file)
        if coordinates is None or features is None:
            print(f"Lỗi đọc dữ liệu")
            return
        
        models = self.data_loader.load_all_models(model_dir)
        if not models:
            print("Lỗi load models")
            return
        
        Path(results_dir).mkdir(exist_ok=True)
        
        for model_name, model in models.items():
            predictions = self.predict(model, features)
            if predictions is None:
                continue
            
            tiff_path = Path(results_dir) / f"{model_name}_prediction_map.tif"
            self.create_tif_grid(coordinates, predictions, tiff_path, model_name)
        
        print("PIPELINE COMPLETED")

def main():
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "model"
    results_dir = base_dir / "results"
    
    data_file = r"/run/media/quan/Quan Vu/25-26_HKI_DATN_QuanVX/train/data/BD_PointGrid_10m_aoi_sample_normalized.csv"
    
    predictor = FloodPredictor()
    predictor.run_prediction_pipeline(data_file, model_dir, results_dir)


if __name__ == "__main__":
    main()