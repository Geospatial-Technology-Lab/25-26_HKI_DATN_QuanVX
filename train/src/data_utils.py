"""Clean data utilities for flood prediction - FINAL VERSION."""

import numpy as np
import geopandas as gpd
import joblib
import gc
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
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


def safe_validate_array(arr: np.ndarray, clip_min: float = 0.0, clip_max: float = 1.0) -> np.ndarray:
    """Utility function để validate và clean array"""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, clip_min, clip_max)


def normalize_with_config(features: np.ndarray, feature_names: list) -> np.ndarray:
    if len(features) == 0:
        return features
    
    # Validate input features first
    features = safe_validate_array(features, -1000.0, 1000.0)
     
    mins = np.array([FEATURE_MIN_MAX[f][0] for f in feature_names])
    maxs = np.array([FEATURE_MIN_MAX[f][1] for f in feature_names])
    ranges = maxs - mins
    
    # Handle zero ranges safely
    ranges = np.where(ranges == 0, 1.0, ranges)
    ranges = np.where(np.isfinite(ranges), ranges, 1.0)
    
    # Normalize: ensure broadcasting works correctly
    if features.shape[1] != len(mins):
        print(f"Warning: Features shape {features.shape} doesn't match config {len(mins)}")
        return np.zeros_like(features)
    
    normalized = (features - mins) / ranges
    return safe_validate_array(normalized)


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
            del gdf; gc.collect()
            return np.array([]), np.array([])
        
        feature_cols = [features_map[f] for f in features_map.keys()]
        valid_indices = gdf[feature_cols].dropna().index
        chunk_clean = gdf.loc[valid_indices, feature_cols]
        
        if len(chunk_clean) == 0:
            del gdf, chunk_clean; gc.collect()
            return np.array([]), np.array([])
        
        coords = np.array([[g.x, g.y] for g in gdf.geometry.loc[valid_indices]])
        coords = safe_validate_array(coords, -180.0, 180.0)
        
        features = normalize_with_config(chunk_clean.values, list(features_map.keys()))
        # Cleanup
        del gdf, chunk_clean, feature_cols, features_map, valid_indices
        gc.collect()
        
        return coords, features
        
    except Exception as e:
        print(f"Chunk {chunk_idx} error: {e}")
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
    
    # Validate input features
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    if features.ndim != 2:
        print(f"Warning: Features shape {features.shape} không đúng format")
        return np.zeros(len(features))
    
    predictions = []
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        try:
            # Input validation
            batch = safe_validate_array(batch, -100.0, 100.0)
            
            pred = model.predict(batch)
            pred = safe_validate_array(pred, -1000.0, 1000.0)
            predictions.append(pred)
        except Exception as e:
            print(f"Prediction error in batch {i}: {e}")
            predictions.append(np.zeros(len(batch)))
        
        # GPU cleanup
        if GPU_AVAILABLE and i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
    
    result = np.concatenate(predictions) if predictions else np.array([])
    return safe_validate_array(result)


def load_models(model_dir: Path) -> dict:
    """Load all models."""
    models = {}
    model_files = list(Path(model_dir).glob("*.joblib")) + list(Path(model_dir).glob("*.pkl"))
    
    for f in tqdm(model_files, desc="Loading models"):
        try:
            models[f.stem] = joblib.load(f)
        except Exception as e:
            print(f"Failed to load {f.name}: {e}")
    
    return models


def predict_to_tiff(models: dict, file_path: Path, layer_name: str, output_dir: Path, 
                   chunk_size: int = 50000, pixel_size: float = 0.00009) -> None:
    
    print("Creating TIFF predictions")
    
    # Test file reading
    try:
        test_gdf = gpd.read_file(str(file_path), layer=layer_name, rows=10)
        features_map = map_features(test_gdf.columns)
        print(f"Data OK, features: {list(features_map.keys())}")
        del test_gdf; gc.collect()
        if not features_map:
            print("No valid features!"); return
    except Exception as e:
        print(f"Data error: {e}"); return
    
    output_dir.mkdir(exist_ok=True)
    
    # Calculate raster from config
    x_min, y_min, x_max, y_max = STUDY_AREA_BOUNDS
    width = int((x_max - x_min) / pixel_size) + 1
    height = int((y_max - y_min) / pixel_size) + 1
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    
    print(f"Raster: {width}x{height} = {width*height:,} pixels")
    estimated_chunks = (TOTAL_ROWS // chunk_size) + 1
    max_workers = max(1, mp.cpu_count() // 2)
    
    # Process each model
    for model_name, model in models.items():
        print(f"\n{model_name}")
        
        # Create raster với giá trị mặc định -9999
        raster = np.full((height, width), -9999.0, dtype=np.float32)
        
        # Process chunks
        chunk_args = [(i, file_path, layer_name, chunk_size) for i in range(estimated_chunks)]
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                chunk_results = list(tqdm(
                    executor.map(process_chunk_parallel, chunk_args),
                    total=len(chunk_args), desc="Loading"
                ))
            
            del chunk_args; gc.collect()
            
            # Process predictions
            total_processed = 0
            for chunk_idx, (coords, features) in tqdm(chunk_results, desc="Predicting"):
                if len(features) == 0: continue
                
                total_processed += len(features)
                
                try:
                    preds = predict_gpu_batch(model, features)
                    preds = safe_validate_array(preds)
                except Exception as e:
                    print(f"Prediction error in chunk {chunk_idx}: {e}")
                    preds = np.zeros(len(features))
                    continue
                
                # Map to pixels
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(np.int32)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(np.int32)
                
                # Check valid indices first
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    # Apply clipping to valid indices only for safety
                    valid_cols = np.clip(cols[valid], 0, width - 1)
                    valid_rows = np.clip(rows[valid], 0, height - 1)
                    preds_valid = preds[valid].astype(np.float32)
                    raster[valid_rows, valid_cols] = preds_valid
                
                del coords, features, preds, cols, rows, valid
                if chunk_idx % 20 == 0: gc.collect()
            
            del chunk_results; gc.collect()
        except Exception as e:
            print(f"Error: {e}")
            # Fallback sequential processing
            for chunk_idx in tqdm(range(estimated_chunks), desc="Sequential"):
                coords, features = load_chunk_clean(file_path, layer_name, chunk_size, chunk_idx)
                if len(features) == 0: continue
                
                try:
                    preds = predict_gpu_batch(model, features)
                    preds = safe_validate_array(preds)
                except Exception as e:
                    print(f"Prediction error in chunk {chunk_idx}: {e}")
                    preds = np.zeros(len(features))
                    continue
                
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(np.int32)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(np.int32)
                
                # Check valid indices first
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    valid_cols = np.clip(cols[valid], 0, width - 1)
                    valid_rows = np.clip(rows[valid], 0, height - 1)
                    preds_valid = preds[valid].astype(np.float32)
                    raster[valid_rows, valid_cols] = preds_valid
                
                del coords, features, preds, cols, rows, valid
                if chunk_idx % 20 == 0: gc.collect()
        
        # Print stats
        valid_pixels = (raster != -9999)
        print(f"Processed: {total_processed:,} points")
        print(f"Valid pixels: {np.sum(valid_pixels):,}/{raster.size:,}")
        if np.any(valid_pixels):
            values = raster[valid_pixels]
            print(f"Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
        
        # Save TIFF với nodata = -9999
        output_path = output_dir / f"{model_name}_flood_probability.tif"
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype=np.float32,
            crs=CRS.from_epsg(4326), transform=transform, nodata=-9999,
            compress='lzw', tiled=True
        ) as dst:
            dst.write(raster, 1)
        
        print(f"Saved: {output_path.name}")
        del raster; gc.collect()
    
    print("Done!")