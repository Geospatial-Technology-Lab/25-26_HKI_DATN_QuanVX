import numpy as np
import pandas as pd
import arcpy
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
from ml_hyper_parameter import get_model_params
from feature_config import FEATURES, FEATURE_MIN_MAX, STUDY_AREA_BOUNDS, TOTAL_ROWS


def map_features(columns: list) -> dict:
    return {col: col for col in columns if col in FEATURES}


def train_models():
    csv_path = "/run/media/quan/Quan Vu/25-26_HKI_DATN_QuanVX/train/data/training_points.csv"
    
    # CSV data is already normalized, use directly
    df = pd.read_csv(csv_path).dropna()
    feature_columns = [col for col in df.columns if col != 'flood']
    X, y = df[feature_columns].values, df['flood'].values
    
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    # Free training data
    del df, X, y, X_train, y_train
    gc.collect()
    
    return models


def normalize_gdb_features(features: np.ndarray, feature_names: list) -> np.ndarray:
    """Normalize raw GDB data using FEATURE_MIN_MAX"""
    if len(features) == 0:
        return features
    
    mins = np.array([FEATURE_MIN_MAX[f][0] for f in feature_names])
    maxs = np.array([FEATURE_MIN_MAX[f][1] for f in feature_names])
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    
    if features.shape[1] != len(mins):
        return np.zeros_like(features)
    
    # Normalize raw GDB data: (value - min) / (max - min)
    normalized = (features - mins) / ranges
    return safe_validate_array(normalized)


def load_chunk_clean(file_path: Path, layer_name: str, chunk_size: int, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load and process raw GDB chunk using arcpy"""
    try:
        start = chunk_idx * chunk_size
        end = start + chunk_size
        
        # Sử dụng arcpy để đọc dữ liệu từ file geodatabase
        feature_class = str(file_path) + "\\" + layer_name
        
        # Lấy danh sách các field
        field_names = [field.name for field in arcpy.ListFields(feature_class) 
                      if field.type not in ['OID', 'Geometry']]
        
        # Lọc các field cần thiết
        features_map = map_features(field_names)
        if not features_map:
            return np.array([]), np.array([])
        
        feature_cols = list(features_map.keys())
        
        # Đọc dữ liệu từ feature class với SearchCursor
        coords_list = []
        features_list = []
        raw_data_debug = []  # Để debug chunk đầu tiên
        
        with arcpy.da.SearchCursor(feature_class, feature_cols + ['SHAPE@XY']) as cursor:
            count = 0
            for row in cursor:
                if count < start:
                    count += 1
                    continue
                if count >= end:
                    break
                
                # Debug: Lưu 20 dòng đầu tiên của chunk đầu tiên
                if chunk_idx == 0 and len(raw_data_debug) < 20:
                    raw_data_debug.append(row)
                
                # Lấy tọa độ từ geometry
                xy = row[-1]  # SHAPE@XY
                if xy is not None:
                    coords_list.append([xy[0], xy[1]])
                    
                    # Lấy giá trị features
                    feature_values = row[:-1]
                    # Kiểm tra và xử lý None values (tưưng đương dropna)
                    if None not in feature_values:
                        features_list.append(feature_values)
                    else:
                        # Loại bỏ nếu có giá trị None
                        coords_list.pop()
                
                count += 1
        
        # Debug: In dữ liệu thô cho chunk đầu tiên
        if chunk_idx == 0 and raw_data_debug:
            print(f"\n=== DEBUG CHUNK ĐẦU TIÊN ({chunk_idx}) ===")
            print(f"20 dòng đầu tiên (dữ liệu thô):")
            for i, row in enumerate(raw_data_debug[:20]):
                print(f"Dòng {i+1}: {row}")
            print(f"\nTổng số dòng thô đọc được: {count-start}")
        
        if len(coords_list) == 0 or len(features_list) == 0:
            if chunk_idx == 0:
                print("Không có dữ liệu hợp lệ sau khi dropna!")
            return np.array([]), np.array([])
        
        coords = np.array(coords_list)
        
        # Normalize raw GDB data using min/max ranges
        features = np.array(features_list)
        features = normalize_gdb_features(features, feature_cols)
        
        # Debug: In dữ liệu sau khi xử lý cho chunk đầu tiên
        if chunk_idx == 0:
            print(f"\nSau khi dropna và chuẩn hóa:")
            print(f"Số dòng còn lại: {len(features)}")
            print(f"Các features: {feature_cols}")
            print(f"5 dòng đầu tiên sau chuẩn hóa:")
            for i in range(min(5, len(features))):
                print(f"Dòng {i+1}: coords={coords[i]}, features={features[i]}")
            print(f"Giá trị min-max của features: min={features.min():.6f}, max={features.max():.6f}")
            print("=== KẾt THÚC DEBUG ===")
        
        return coords, features
        
    except Exception as e:
        print(f"Lỗi khi đọc chunk {chunk_idx}: {e}")
        gc.collect()
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
    
    # Test file reading với arcpy
    try:
        feature_class = str(file_path) + "\\" + layer_name
        
        # Kiểm tra xem feature class có tồn tại không
        if not arcpy.Exists(feature_class):
            print(f"Feature class không tồn tại: {feature_class}")
            return
        
        # Lấy danh sách các field
        field_names = [field.name for field in arcpy.ListFields(feature_class) 
                      if field.type not in ['OID', 'Geometry']]
        
        features_map = map_features(field_names)
        if not features_map:
            print("Không có feature hợp lệ!")
            return
            
    except Exception as e:
        print(f"Lỗi dữ liệu: {e}")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Calculate raster from config
    x_min, y_min, x_max, y_max = STUDY_AREA_BOUNDS
    width = int((x_max - x_min) / pixel_size) + 1
    height = int((y_max - y_min) / pixel_size) + 1
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    
    estimated_chunks = (TOTAL_ROWS // chunk_size) + 1
    max_workers = max(1, mp.cpu_count() // 2)
    
    # Process each model
    for model_name, model in models.items():
        print(f"\n{model_name}")
        
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
                except Exception:
                    preds = np.zeros(len(features))
                    continue
                
                # Map to pixels
                cols = ((coords[:, 0] - x_min) / pixel_size).astype(np.int32)
                rows = ((y_max - coords[:, 1]) / pixel_size).astype(np.int32)
                
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                
                if valid.any():
                    valid_cols = np.clip(cols[valid], 0, width - 1)
                    valid_rows = np.clip(rows[valid], 0, height - 1)
                    preds_valid = preds[valid].astype(np.float32)
                    raster[valid_rows, valid_cols] = preds_valid
                
                # Clean up after each chunk
                del coords, features, preds, cols, rows, valid, valid_cols, valid_rows, preds_valid
                if chunk_idx % 10 == 0: gc.collect()  # More frequent cleanup
            
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
                
                # Clean up after each chunk
                del coords, features, preds, cols, rows, valid, valid_cols, valid_rows, preds_valid
                if chunk_idx % 10 == 0: gc.collect()  # More frequent cleanup
        
        # Save TIFF
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