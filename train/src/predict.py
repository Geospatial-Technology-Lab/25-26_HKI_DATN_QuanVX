import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

from data_utils import load_and_process_geodata, load_all_models
from gpu import GPUOptimizer, optimize_for_big_data


class FloodPredictor:
    """Unified flood prediction pipeline"""
    
    def __init__(self, auto_optimize=True):
        self.pixel_size = 0.00009
        self.crs = CRS.from_epsg(4326)
        self.auto_optimize = auto_optimize
        self.gpu_optimizer = None

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
        
        # Điền giá trị vào grid
        for i, (x, y) in enumerate(coordinates):
            col = int((x - min_x) / self.pixel_size)
            row = int((max_y - y) / self.pixel_size)
            
            if 0 <= row < height and 0 <= col < width:
                grid[row, col] = predictions[i]
        
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
            compress='lzw'
        ) as dst:
            dst.write(grid, 1)

    def run_prediction_pipeline(self, data_file, model_dir, results_dir, layer_name=None):
        """Unified prediction pipeline - single workflow for all data sizes"""
        
        print("🔄 Đang phân tích dữ liệu lưới điểm...")
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
        
        # Xử lý từng mô hình
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n🔮 [{i}/{len(models)}] Bắt đầu dự đoán với mô hình: {model_name}")
            tiff_path = results_path / f"{model_name}_prediction_map.tif"
            
            if self.gpu_optimizer:
                # Sử dụng GPU optimization cho big data
                success = self.gpu_optimizer.process_model_parallel(
                    model, features, coordinates, model_name, tiff_path, 
                    self.pixel_size, self.crs
                )
                if not success:
                    print(f"⚠️ GPU optimization thất bại, fallback về CPU...")
                    predictions = self.predict(model, features)
                    self.create_tif_grid(coordinates, predictions, tiff_path)
            else:
                # Sử dụng CPU processing cho dữ liệu nhỏ
                predictions = self.predict(model, features)
                self.create_tif_grid(coordinates, predictions, tiff_path)
            
            print(f"✅ Hoàn thành {model_name} - Lưu tại: {tiff_path.name}")


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
    predictor = FloodPredictor()
    predictor.run_prediction_pipeline(data_file, model_dir, results_dir, layer_name)
    print("\n🎉 Hoàn thành tất cả dự đoán! Kết quả đã được lưu trong thư mục results.")


if __name__ == "__main__":
    main()