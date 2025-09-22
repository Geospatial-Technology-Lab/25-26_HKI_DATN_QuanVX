import numpy as np
import pandas as pd
import geopandas as gpd
import gc
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from math import sqrt
import fiona
from ml_hyper_parameter import get_model_params
from feature_config import FEATURES, FEATURE_MIN_MAX, STUDY_AREA_BOUNDS, TOTAL_ROWS


def safe_validate_array(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def map_features(columns: list) -> dict:
    return {col: col for col in columns if col in FEATURES}


def train_models():
    csv_path = r"Z:\guest01\QuanVX\25-26_HKI_DATN_QuanVX\train\data\training_points.csv"
    
    df = pd.read_csv(csv_path).dropna()
    feature_columns = [col for col in df.columns if col != 'flood']
    X, y = df[feature_columns].values, df['flood'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    models_to_train = [
        ('rf', 'pso'), ('rf', 'po'), ('rf', 'rso'),
        ('svm', 'pso'), ('svm', 'po'), ('svm', 'rso'),
        ('xgb', 'pso'), ('xgb', 'po'), ('xgb', 'rso')
    ]
    
    print("Training models with CSV data (already normalized)...")
    for model_type, optimization_method in models_to_train:
        params = get_model_params(model_type, optimization_method)
        if not params:
            continue
        
        if model_type == 'rf':
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif model_type == 'svm':
            model = SVR(**params)
        elif model_type == 'xgb':
            model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        else:
            continue
        
        model.fit(X_train, y_train)
        model_name = f"{optimization_method}_{model_type}"
        models[model_name] = model
        print(f"Trained {model_name}")

        y_pred = model.predict(X_test)

        rmse = sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = model.score(X_test, y_test)

        print (f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
    
    del df, X, y, X_train, y_train
    gc.collect()
    
    return models


def normalize_gdb_features(features: np.ndarray, feature_names: list) -> np.ndarray:
    if len(features) == 0:
        return features
    
    mins = np.array([FEATURE_MIN_MAX[f][0] for f in feature_names])
    maxs = np.array([FEATURE_MIN_MAX[f][1] for f in feature_names])
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    
    if features.shape[1] != len(mins):
        return np.zeros_like(features)
    
    normalized = (features - mins) / ranges
    return safe_validate_array(normalized)


def load_chunk_clean(file_path: Path, layer_name: str, chunk_size: int, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        start = chunk_idx * chunk_size
        end = start + chunk_size
        
        gdb_path = str(file_path)
        
        try:
            # Use fiona for streaming read - only read the chunk we need
            coords_list = []
            features_list = []
            
            with fiona.open(gdb_path, layer=layer_name) as src:
                # Get feature schema info only once
                if chunk_idx == 0:
                    print(f"\n=== CHUNK {chunk_idx}: DỮ LIỆU THÔ ===\n")
                    print(f"Total features in layer: {len(src)}")
                    print(f"Properties: {list(src.schema['properties'].keys())}")
                    print(f"\n50 dòng đầu tiên (dữ liệu thô):")
                
                # Skip to start position and read only chunk_size records
                current_idx = 0
                records_read = 0
                raw_data_for_display = []  # Store raw data for display
                
                for feature in src:
                    if current_idx < start:
                        current_idx += 1
                        continue
                    
                    if records_read >= chunk_size:
                        break
                    
                    # Store raw data for display (first 50 records)
                    if chunk_idx == 0 and records_read < 50:
                        geom_info = f"geom_type={feature['geometry']['type']}" if feature['geometry'] else "geometry=None"
                        props_info = ", ".join([f"{k}={v}" for k, v in list(feature['properties'].items())[:5]])
                        raw_data_for_display.append(f"Record {records_read}: {geom_info}, {props_info}...")
                    
                    # Process this feature
                    geom = feature['geometry']
                    props = feature['properties']
                    
                    if geom and geom['type'] in ['Point', 'Polygon', 'MultiPolygon']:
                        # Extract coordinates
                        if geom['type'] == 'Point':
                            coords = geom['coordinates']
                        else:
                            # Get centroid for polygons
                            import shapely.geometry as sg
                            shape = sg.shape(geom)
                            centroid = shape.centroid
                            coords = [centroid.x, centroid.y]
                        
                        coords_list.append(coords)
                        
                        # Extract numeric features
                        feature_values = []
                        for col in FEATURES:
                            if col in props:
                                val = props[col]
                                if val is not None and not pd.isna(val):
                                    feature_values.append(float(val))
                                else:
                                    feature_values.append(0.0)
                            else:
                                feature_values.append(0.0)
                        
                        features_list.append(feature_values)
                    
                    records_read += 1
                    current_idx += 1
                
                # Print raw data for first chunk
                if chunk_idx == 0:
                    for raw_line in raw_data_for_display:
                        print(raw_line)
            
            if len(coords_list) == 0 or len(features_list) == 0:
                return np.array([]), np.array([])
            
            coords = np.array(coords_list)
            features = np.array(features_list)
            
            # Normalize features
            if len(features) > 0:
                features = normalize_gdb_features(features, FEATURES)
            
            # Debug info for first chunk only
            if chunk_idx == 0 and len(coords) > 0:
                print(f"\n=== CHUNK {chunk_idx}: DỮ LIỆU SAU TIỀN XỬ LÝ ===")
                print(f"Số dòng còn lại sau lọc: {len(coords)}")
                print(f"Các feature được sử dụng: {FEATURES}")
                print(f"Shape của coords: {coords.shape}")
                print(f"Shape của features: {features.shape}")
                
                print(f"\nDữ liệu đã được chuẩn hóa (50 dòng đầu):")
                max_display = min(50, len(coords))
                for i in range(max_display):
                    coord_str = f"coords=({coords[i][0]:.6f}, {coords[i][1]:.6f})"
                    feature_str = ", ".join([f"{FEATURES[j]}={features[i][j]:.4f}" for j in range(min(5, len(FEATURES)))])
                    print(f"Row {i}: {coord_str}, features=[{feature_str}...]")
                
                if len(coords) > 50:
                    print(f"... và {len(coords) - 50} dòng khác")
                print("=" * 50)
            
            return coords, features
        except Exception as e:
            print(f"Error reading geodatabase chunk {chunk_idx}: {e}")
            return np.array([]), np.array([])


def process_chunk_parallel(chunk_args):
    chunk_idx, file_path, layer_name, chunk_size = chunk_args
    return chunk_idx, load_chunk_clean(file_path, layer_name, chunk_size, chunk_idx)


def predict_gpu_batch(model, features, batch_size=8000):
    if len(features) == 0:
        return np.array([])
    
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    if features.ndim != 2:
        return np.zeros(len(features))
    
    predictions = []
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        
        try:
            pred = model.predict(batch)
            predictions.append(pred)
        except Exception:
            predictions.append(np.zeros(len(batch)))
        
        # Clean batch after each iteration
        del batch
    
    result = np.concatenate(predictions) if predictions else np.array([])
    del predictions
    return result


def predict_to_tiff(file_path: Path, layer_name: str, output_dir: Path, 
                   chunk_size: int = 50000, pixel_size: float = 0.00009) -> None:
    
    print("Training models...")
    models = train_models()
    print(f"Trained {len(models)} models")
    
    print("Creating TIFF predictions")
    
    try:
        gdb_path = str(file_path)
        gdf = gpd.read_file(gdb_path, layer=layer_name, rows=1)
        
        numeric_columns = [col for col in gdf.columns 
                          if col != 'geometry' and gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        features_map = map_features(numeric_columns)
        if not features_map:
            print("Không có feature hợp lệ!")
            return
            
    except Exception as e:
        print(f"Lỗi dữ liệu: {e}")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    x_min, y_min, x_max, y_max = STUDY_AREA_BOUNDS
    width = int((x_max - x_min) / pixel_size) + 1
    height = int((y_max - y_min) / pixel_size) + 1
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    
    estimated_chunks = (TOTAL_ROWS // chunk_size) + 1
    max_workers = max(1, mp.cpu_count() // 2)
    
    for model_name, model in models.items():
        print(f"\n{model_name}")
        
        raster = np.full((height, width), -9999.0, dtype=np.float32)
        
        chunk_args = [(i, file_path, layer_name, chunk_size) for i in range(estimated_chunks)]
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                chunk_results = list(tqdm(
                    executor.map(process_chunk_parallel, chunk_args),
                    total=len(chunk_args), desc="Loading"
                ))
            
            del chunk_args; gc.collect()
            
            total_processed = 0
            for chunk_idx, (coords, features) in tqdm(chunk_results, desc="Predicting"):
                if len(features) == 0: continue
                
                total_processed += len(features)
                
                try:
                    preds = predict_gpu_batch(model, features)
                except Exception:
                    preds = np.zeros(len(features))
                    continue
                
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(np.int32)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(np.int32)
                
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    valid_cols = np.clip(cols[valid], 0, width - 1)
                    valid_rows = np.clip(rows[valid], 0, height - 1)
                    preds_valid = preds[valid].astype(np.float32)
                    raster[valid_rows, valid_cols] = preds_valid
                
                del coords, features, preds, cols, rows, valid, valid_cols, valid_rows, preds_valid
                if chunk_idx % 10 == 0: gc.collect()
            
            del chunk_results; gc.collect()
        except Exception:
            for chunk_idx in tqdm(range(estimated_chunks), desc="Sequential"):
                coords, features = load_chunk_clean(file_path, layer_name, chunk_size, chunk_idx)
                if len(features) == 0: continue
                
                try:
                    preds = predict_gpu_batch(model, features)
                except Exception:
                    preds = np.zeros(len(features))
                    continue
                
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(np.int32)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(np.int32)
                
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    valid_cols = np.clip(cols[valid], 0, width - 1)
                    valid_rows = np.clip(rows[valid], 0, height - 1)
                    preds_valid = preds[valid].astype(np.float32)
                    raster[valid_rows, valid_cols] = preds_valid
                
                del coords, features, preds, cols, rows, valid, valid_cols, valid_rows, preds_valid
                if chunk_idx % 10 == 0: gc.collect()
        
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
