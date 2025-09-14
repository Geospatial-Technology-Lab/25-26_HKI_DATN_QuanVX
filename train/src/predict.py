"""Simple flood prediction with chunked processing."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm
import gc

from data_utils import load_sample, load_chunk, load_models, get_row_count


class FloodPredictor:
    """Simple flood prediction."""
    
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
        self.pixel_size = 0.00009
        self.crs = CRS.from_epsg(4326)
        self.grids = {}
    
    def init_grids(self, sample_gdf, model_names: list):
        """Setup grids."""
        coords = np.array([[g.x, g.y] for g in sample_gdf.geometry])
        bounds = (coords[:, 0].min() - 0.01, coords[:, 1].min() - 0.01,
                 coords[:, 0].max() + 0.01, coords[:, 1].max() + 0.01)
        
        w = int((bounds[2] - bounds[0]) / self.pixel_size) + 1
        h = int((bounds[3] - bounds[1]) / self.pixel_size) + 1
        
        for name in model_names:
            self.grids[name] = {
                'grid': np.full((h, w), np.nan, dtype=np.float32),
                'bounds': bounds,
                'transform': from_bounds(*bounds, w, h)
            }
        print(f"📐 Grid: {w}x{h}")
    
    def update_grid(self, model_name: str, coords: np.ndarray, preds: np.ndarray):
        """Update grid with predictions."""
        if model_name not in self.grids:
            return
        
        grid_info = self.grids[model_name]
        grid = grid_info['grid']
        bounds = grid_info['bounds']
        h, w = grid.shape
        
        for i, (x, y) in enumerate(coords):
            col = int((x - bounds[0]) / self.pixel_size)
            row = int((bounds[3] - y) / self.pixel_size)
            if 0 <= row < h and 0 <= col < w:
                grid[row, col] = preds[i]
    
    def save_grid(self, model_name: str, output_path: Path):
        """Save grid to TIFF."""
        if model_name not in self.grids:
            return
        
        info = self.grids[model_name]
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=info['grid'].shape[0], width=info['grid'].shape[1],
            count=1, dtype=np.float32, crs=self.crs,
            transform=info['transform'], nodata=np.nan,
            compress='lzw', tiled=True
        ) as dst:
            dst.write(info['grid'], 1)
    
    def predict(self, data_file: str, model_dir: str, results_dir: str, layer_name: str = None):
        """Main prediction pipeline."""
        print("🚀 Simple Flood Prediction")
        
        data_path = Path(data_file)
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        # Load sample and models
        sample = load_sample(data_path, layer_name, 1000)
        if not sample:
            raise ValueError("Cannot load sample")
        
        models = load_models(Path(model_dir))
        total_rows = get_row_count(data_path, layer_name)
        num_chunks = (total_rows // self.chunk_size) + 1
        
        print(f"📊 {total_rows:,} rows, {num_chunks} chunks, {len(models)} models")
        
        # Setup grids
        self.init_grids(sample['gdf'], list(models.keys()))
        
        # Process chunks
        for chunk_idx in tqdm(range(num_chunks), desc="Processing"):
            coords, features = load_chunk(data_path, layer_name, self.chunk_size, chunk_idx)
            
            if len(coords) == 0:
                continue
            
            for model_name, model in models.items():
                preds = model.predict(features)
                self.update_grid(model_name, coords, preds)
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        # Save results
        for model_name in models.keys():
            output_path = results_path / f"{model_name}_prediction.tif"
            self.save_grid(model_name, output_path)
            print(f"✅ {output_path.name}")
        
        print("🎉 Done!")


def main():
    """Run prediction."""
    predictor = FloodPredictor(chunk_size=100_000)
    predictor.predict(
        data_file=r"D:\QuanVX\QuanVX\Default.gdb\a000000c8.gdbtable",
        model_dir=Path(__file__).parent.parent / "model",
        results_dir=Path(__file__).parent.parent / "results",
        layer_name="RasterT_Extract1"
    )


if __name__ == "__main__":
    main()