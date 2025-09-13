import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from load_data import DataLoader

class FloodPredictor:
    def __init__(self):
        self.pixel_size = 0.00009
        self.crs = CRS.from_epsg(4326)
        self.data_loader = DataLoader()
        
    def load_data(self, data_file):
        return self.data_loader.load_data(data_file)
    
    def predict(self, model, features):
        return model.predict(features)
    
    def create_tif_grid(self, coordinates, predictions, output_path):
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
                         count=1, dtype=np.float32, crs=self.crs, transform=transform, 
                         nodata=np.nan, compress='lzw') as dst:
            dst.write(grid, 1)
    
    def run_prediction_pipeline(self, data_file, model_dir, results_dir):
        print("🔄 Đang tải dữ liệu...")
        coordinates, features = self.load_data(data_file)
        print(f"✅ Tải dữ liệu thành công! ({len(coordinates)} điểm dữ liệu)")
        
        print("🔄 Đang tải các mô hình...")
        models = self.data_loader.load_all_models(model_dir)
        print(f"✅ Tải {len(models)} mô hình thành công!")
        
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        # Process multiple models
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n🔮 [{i}/{len(models)}] Bắt đầu dự đoán với mô hình: {model_name}")
            predictions = self.predict(model, features)
            tiff_path = results_path / f"{model_name}_prediction_map.tif"
            self.create_tif_grid(coordinates, predictions, tiff_path)
            print(f"✅ Hoàn thành {model_name} - Lưu tại: {tiff_path.name}")

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "data" / "BD_PointGrid_10m_aoi_sample.gpd"
    model_dir = base_dir / "model"
    results_dir = base_dir / "results"
    
    print("🚀 Bắt đầu quá trình dự đoán lũ lụt...")
    predictor = FloodPredictor()
    predictor.run_prediction_pipeline(data_file, model_dir, results_dir)
    print("\n🎉 Hoàn thành tất cả dự đoán! Kết quả đã được lưu trong thư mục results.")

if __name__ == "__main__":
    main()