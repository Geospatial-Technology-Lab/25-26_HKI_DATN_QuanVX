import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm
import gc

from data_utils import load_and_process_geodata, load_all_models
from gpu import GPUOptimizer, optimize_for_big_data


class FloodPredictor:
    """Optimized flood prediction pipeline for big data"""
    
    def __init__(self, chunk_size=1_000_000):
        self.pixel_size = 0.00009
        self.crs = CRS.from_epsg(4326)
        self.chunk_size = chunk_size
        self.auto_optimize = True
        self.gpu_optimizer = None

    def predict_chunked(self, model, features, model_name):
        """Make predictions in chunks to avoid memory overflow"""
        total_samples = len(features)
        predictions = np.zeros(total_samples, dtype=np.float32)
        
        num_chunks = (total_samples // self.chunk_size) + 1
        
        with tqdm(total=num_chunks, desc=f"Predicting {model_name}") as pbar:
            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, total_samples)
                
                chunk_features = features[start_idx:end_idx]
                predictions[start_idx:end_idx] = model.predict(chunk_features)
                
                pbar.update(1)
                
                # Memory cleanup
                if i % 5 == 0:
                    gc.collect()
        
        return predictions

    def predict(self, model, features):
        """Make predictions using the model"""
        return model.predict(features)
    
    def create_tif_grid(self, coordinates, predictions, output_path):

        # Tính toán bounding box
        min_x, max_x = coordinates[:, 0].min(), coordinates[:, 0].max()
        min_y, max_y = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        # Tính kích thước grid
        width = int((max_x - min_x) / self.pixel_size) + 1
        height = int((max_y - min_y) / self.pixel_size) + 1
        
        # Khởi tạo grid với giá trị NaN
        grid = np.full((height, width), np.nan, dtype=np.float32)
        
        # Điền giá trị vào grid với thanh tiến trình
        with tqdm(total=len(coordinates), desc="Creating grid") as pbar:
            for i in range(0, len(coordinates), 10000):  # Process in batches
                batch_end = min(i + 10000, len(coordinates))
                
                for j in range(i, batch_end):
                    x, y = coordinates[j]
                    col = int((x - min_x) / self.pixel_size)
                    row = int((max_y - y) / self.pixel_size)
                    
                    if 0 <= row < height and 0 <= col < width:
                        grid[row, col] = predictions[j]
                
                pbar.update(batch_end - i)
        
        # Tạo transform cho GeoTIFF
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        
        # Lưu file TIFF
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=self.crs,
            transform=transform,
            nodata=np.nan,
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512
        ) as dst:
            dst.write(grid, 1)

    def run_prediction_pipeline(self, data_file, model_dir, results_dir, layer_name=None):
        """Optimized pipeline for big data processing"""
        
        print("� Starting Big Data Flood Prediction Pipeline")
        print(f"⚙️ Chunk size: {self.chunk_size:,} points")
        print("📊 Sử dụng luồng xử lý thống nhất cho tất cả kích thước dữ liệu")
        if layer_name:
            print(f"🎯 Sử dụng layer: {layer_name}")
        print(f"🔍 Sẽ tải 13 features (không bao gồm target 'flood')")
        
        # Load data using unified data processing
        print("🚀 Sử dụng data processing thống nhất...")
        coordinates, features = load_and_process_geodata(Path(data_file), layer_name)
        
        if len(coordinates) == 0:
            print("❌ Không thể load dữ liệu!")
            return
        
        # Auto-configure optimization based on data size
        if self.auto_optimize and len(coordinates) > 100000:  # 100k threshold
            config = optimize_for_big_data(len(coordinates))
            self.gpu_optimizer = GPUOptimizer(**config)
            print(f"🚀 Kích hoạt GPU Optimization cho {len(coordinates):,} điểm")
        
        print(f"✅ Tải dữ liệu thành công! ({len(coordinates):,} điểm dữ liệu)")
        
        # Load models using unified loader
        print("🔄 Đang tải các mô hình...")
        models = load_all_models(model_dir)
        print(f"✅ Tải {len(models)} mô hình thành công!")
        
        # Tạo thư mục kết quả
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        # Xử lý từng mô hình với thanh tiến trình
        print(f"\n🔮 Processing {len(models)} models...")
        
        for model_name, model in tqdm(models.items(), desc="Models"):
            print(f"\n� Processing: {model_name}")
            
            # Use chunked prediction for large datasets
            if len(coordinates) > 1_000_000:  # 1M+ points
                predictions = self.predict_chunked(model, features, model_name)
            else:
                predictions = self.predict(model, features)
            
            # Create TIF
            tiff_path = results_path / f"{model_name}_prediction.tif"
            self.create_tif_grid(coordinates, predictions, tiff_path)
            
            print(f"✅ Saved: {tiff_path.name}")
            
            # Memory cleanup
            del predictions
            gc.collect()


def main():
    """
    Hàm main chạy toàn bộ pipeline dự đoán
    """
    # Thiết lập đường dẫn
    base_dir = Path(__file__).parent.parent
    # Đường dẫn file GDB Windows
    data_file = r"D:\QuanVX\QuanVX\Default.gdb\a000000c8.gdbtable"
    layer_name = "RasterT_Extract1"  # Tên layer trong ArcGIS Pro
    model_dir = base_dir / "model"
    results_dir = base_dir / "results"
    
    # Bắt đầu quá trình dự đoán
    print("🚀 Bắt đầu quá trình dự đoán lũ lụt...")
    print(f"📁 File GDB: {data_file}")
    print(f"🎯 Sử dụng layer: {layer_name}")
    print(f"📋 Input: 13 features (không có target 'flood')")
    
    predictor = FloodPredictor(chunk_size=1_000_000)  # 1M chunk size
    predictor.run_prediction_pipeline(data_file, model_dir, results_dir, layer_name)
    print("\n🎉 Hoàn thành tất cả dự đoán! Kết quả đã được lưu trong thư mục results.")


if __name__ == "__main__":
    main()