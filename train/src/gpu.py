import numpy as np
import rasterio
from rasterio.transform import from_bounds
import gc
import psutil
from pathlib import Path
from tqdm import tqdm

class GPUOptimizer:
    
    def __init__(self, batch_size=500000, tile_size=5000):
        self.batch_size = batch_size
        self.tile_size = tile_size
    
    def predict_batched(self, model, features: np.ndarray) -> np.ndarray:
        n_samples = len(features)
        predictions = []
        
        print(f"🚀 Batch Processing: {n_samples:,} điểm")
        
        with tqdm(total=n_samples, desc="GPU Prediction") as pbar:
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                batch_predictions = model.predict(features[i:end_idx])
                predictions.extend(batch_predictions)
                
                pbar.update(end_idx - i)
                
                if i % (self.batch_size * 5) == 0:
                    memory_gb = psutil.virtual_memory().used / (1024**3)
                    pbar.set_postfix({"RAM": f"{memory_gb:.1f}GB"})
                    gc.collect()
        
        return np.array(predictions)
    
    def create_tiled_tif(self, coordinates: np.ndarray, predictions: np.ndarray, 
                        output_path: Path, pixel_size: float, crs):
        min_x, max_x = coordinates[:, 0].min(), coordinates[:, 0].max()
        min_y, max_y = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        width = int((max_x - min_x) / pixel_size) + 1
        height = int((max_y - min_y) / pixel_size) + 1
        
        print(f"🗺️ TIF: {width:,}x{height:,} pixels")
        
        # Tạo lookup dict
        coord_dict = {}
        for i, (x, y) in enumerate(coordinates):
            col = int((x - min_x) / pixel_size)
            row = int((max_y - y) / pixel_size)
            if 0 <= row < height and 0 <= col < width:
                coord_dict[(row, col)] = predictions[i]
        
        # Tạo TIF
        with rasterio.open(
            output_path, 'w', driver='GTiff', height=height, width=width,
            count=1, dtype=np.float32, crs=crs,
            transform=from_bounds(min_x, min_y, max_x, max_y, width, height),
            nodata=np.nan, compress='lzw', tiled=True, blockxsize=512, blockysize=512
        ) as dst:
            for tile_y in range(0, height, self.tile_size):
                for tile_x in range(0, width, self.tile_size):
                    th = min(self.tile_size, height - tile_y)
                    tw = min(self.tile_size, width - tile_x)
                    
                    tile_data = np.full((th, tw), np.nan, dtype=np.float32)
                    for r in range(th):
                        for c in range(tw):
                            key = (tile_y + r, tile_x + c)
                            if key in coord_dict:
                                tile_data[r, c] = coord_dict[key]
                    
                    dst.write(tile_data, 1, window=rasterio.windows.Window(tile_x, tile_y, tw, th))
    
    def process_model_parallel(self, model, features, coordinates, model_name, output_path, pixel_size, crs):
        try:
            print(f"🚀 GPU Processing: {model_name}")
            
            # Dự đoán với batching
            predictions = self.predict_batched(model, features)
            
            # Tạo TIF với tiling
            self.create_tiled_tif(coordinates, predictions, output_path, pixel_size, crs)
            
            print(f"✅ Hoàn thành {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ GPU processing thất bại cho {model_name}: {e}")
            return False

def optimize_for_big_data(data_size: int) -> dict:
    """Cấu hình tối ưu theo data size"""
    if data_size < 1e6:
        return {'batch_size': 100000, 'tile_size': 2000}
    elif data_size < 100e6:
        return {'batch_size': 500000, 'tile_size': 5000}
    else:
        return {'batch_size': 1000000, 'tile_size': 10000}