"""Clean data utilities for flood prediction - FINAL VERSION."""

import numpy as np
import geopandas as gpd
import joblib
import gc
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from feature_config import FEATURES, FEATURE_MIN_MAX, STUDY_AREA_BOUNDS, TOTAL_ROWS

# GPU support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = 'cpu'


def normalize_with_config(features: np.ndarray, feature_names: list) -> np.ndarray:
    if len(features) == 0:
        return features
     
    mins = np.array([FEATURE_MIN_MAX[f][0] for f in feature_names])
    maxs = np.array([FEATURE_MIN_MAX[f][1] for f in feature_names])
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    return (features - mins) / ranges


def map_features(columns: list) -> dict:
    """Map columns to standard features."""
    mapped = {}
    for col in columns:
        if col in FEATURES:
            mapped[col] = col
    return mapped


def load_chunk_clean(file_path: Path, layer_name: str, chunk_size: int, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load chunk, drop nulls, normalize."""
    try:
        start = chunk_idx * chunk_size
        gdf = gpd.read_file(str(file_path), layer=layer_name, rows=chunk_size, skip=start)
        
        if len(gdf) == 0:
            return np.array([]), np.array([])
        
        features_map = map_features(gdf.columns)
        if not features_map:
            # Clean up before return
            del gdf
            gc.collect()
            return np.array([]), np.array([])
        
        feature_cols = [features_map[f] for f in features_map.keys()]
        chunk_clean = gdf[feature_cols].dropna()
        
        if len(chunk_clean) == 0:
            # Clean up before return
            del gdf, chunk_clean
            gc.collect()
            return np.array([]), np.array([])
        
        coords = np.array([[g.x, g.y] for g in gdf.geometry.iloc[:len(chunk_clean)]])
        features = normalize_with_config(chunk_clean.values, list(features_map.keys()))
        
        # Aggressive cleanup
        del gdf, chunk_clean, feature_cols, features_map
        gc.collect()
        
        return coords, features
        
    except Exception as e:
        print(f"❌ Chunk {chunk_idx} error: {e}")
        # Force cleanup on error
        gc.collect()
        return np.array([]), np.array([])


def process_chunk_parallel(chunk_args):
    """Wrapper function for parallel chunk processing."""
    chunk_idx, file_path, layer_name, chunk_size = chunk_args
    return chunk_idx, load_chunk_clean(file_path, layer_name, chunk_size, chunk_idx)


def predict_gpu_batch(model, features, batch_size=8000):
    """GPU prediction với batch để tránh OOM."""
    if len(features) == 0:
        return np.array([])
    
    predictions = []
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        pred = model.predict(batch)
        predictions.append(pred)
        
        # GPU cleanup mỗi batch
        if GPU_AVAILABLE and i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
    
    return np.concatenate(predictions) if predictions else np.array([])


def load_models(model_dir: Path) -> dict:
    """Load all models."""
    models = {}
    model_files = list(Path(model_dir).glob("*.joblib")) + list(Path(model_dir).glob("*.pkl"))
    
    for f in tqdm(model_files, desc="Loading models"):
        try:
            models[f.stem] = joblib.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load {f.name}: {e}")
    
    return models


def predict_to_tiff(models: dict, file_path: Path, layer_name: str, output_dir: Path, 
                   chunk_size: int = 50000, pixel_size: float = 0.00009) -> None:
    
    print(f"🚀 Creating TIFF predictions")
    print(f"🔧 Using device: {DEVICE}")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate raster from config
    x_min, y_min, x_max, y_max = STUDY_AREA_BOUNDS
    width = int((x_max - x_min) / pixel_size) + 1
    height = int((y_max - y_min) / pixel_size) + 1
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    
    print(f"📊 Raster: {width}x{height} = {width*height:,} pixels")
    print(f"📊 Using configured row count: {TOTAL_ROWS:,}")
    
    estimated_chunks = (TOTAL_ROWS // chunk_size) + 1
    
    # Calculate optimal number of workers (use half of CPU cores to avoid overload)
    max_workers = max(1, mp.cpu_count() // 2)
    print(f"🔧 Using {max_workers} parallel workers for chunk processing")
    
    # Process each model sequentially (one at a time)
    for model_name, model in models.items():
        print(f"\n🔮 Processing: {model_name}")
        
        # Create raster array
        raster = np.full((height, width), np.nan, dtype=np.float32)
        
        # Prepare chunk arguments for parallel processing
        chunk_args = [(i, file_path, layer_name, chunk_size) for i in range(estimated_chunks)]
        
        # Process chunks in parallel
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use tqdm to show progress
                chunk_results = list(tqdm(
                    executor.map(process_chunk_parallel, chunk_args),
                    total=len(chunk_args),
                    desc=f"{model_name} - Loading chunks"
                ))
            
            # Clean up chunk arguments to free memory
            del chunk_args
            gc.collect()
            
            # Process predictions for each chunk
            for idx, (chunk_idx, (coords, features)) in enumerate(tqdm(chunk_results, desc=f"{model_name} - Predicting")):
                if len(features) == 0:
                    continue
                
                # Predict
                try:
                    preds = np.clip(predict_gpu_batch(model, features), 0.00, 1.00)
                except:
                    continue
                
                # Map to pixels
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(int)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(int)
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    raster[rows[valid], cols[valid]] = preds[valid]
                
                # Aggressive memory cleanup after each chunk
                del coords, features, preds, cols, rows, valid
                
                # Force garbage collection every 10 chunks
                if idx % 10 == 0:
                    gc.collect()
                    if GPU_AVAILABLE:
                        torch.cuda.empty_cache()
            
            # Final cleanup after all chunks processed
            del chunk_results
            gc.collect()
                        
        except Exception as e:
            print(f"❌ Error in parallel processing for {model_name}: {e}")
            print("🔄 Falling back to sequential processing...")
            
            # Clean up any existing data before fallback
            if 'chunk_results' in locals():
                del chunk_results
            if 'chunk_args' in locals():
                del chunk_args
            gc.collect()
            
            # Fallback to sequential processing if parallel fails
            for chunk_idx in tqdm(range(estimated_chunks), desc=f"{model_name} - Sequential"):
                coords, features = load_chunk_clean(file_path, layer_name, chunk_size, chunk_idx)
                
                if len(features) == 0:
                    continue
                
                # Predict
                try:
                    preds = np.clip(predict_gpu_batch(model, features), 0.00, 1.00)
                except:
                    continue
                
                # Map to pixels
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(int)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(int)
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    raster[rows[valid], cols[valid]] = preds[valid]
                
                # Aggressive cleanup in sequential mode too
                del coords, features, preds, cols, rows, valid
                
                # More frequent cleanup in sequential mode
                if chunk_idx % 10 == 0:
                    gc.collect()
                    if GPU_AVAILABLE:
                        torch.cuda.empty_cache()
        
        # Save TIFF
        output_path = output_dir / f"{model_name}_flood_probability.tif"
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype=np.float32,
            crs=CRS.from_epsg(4326), transform=transform, nodata=np.nan,
            compress='lzw', tiled=True
        ) as dst:
            dst.write(raster, 1)
        
        print(f"✅ Saved: {output_path.name}")
        
        # Aggressive cleanup after each model
        del raster
        gc.collect()
        
        # Additional GPU cleanup if available
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        
        print(f"🧹 Memory cleaned after {model_name}")
    
    # Final cleanup
    print("🧹 Final memory cleanup...")
    gc.collect()
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
    
    print("🎉 All done!")