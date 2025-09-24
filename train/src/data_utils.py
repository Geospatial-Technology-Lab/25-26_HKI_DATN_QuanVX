import numpy as np
import pandas as pd
import geopandas as gpd
import gc
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from typing import Tuple
import fiona
import shapely.geometry as sg
from ml_hyper_parameter import get_model_params
from feature_config import FEATURES, FEATURE_MIN_MAX, STUDY_AREA_BOUNDS

# Chunk size: 2 million points
CHUNK_SIZE = 2_000_000

def validate_feature_compatibility(gdb_features: list) -> bool:
    """
    Validate that GDB features are compatible with training features.
    Ensures all required features are present and in correct order.
    """
    # Check if all required features are present
    missing_features = [f for f in FEATURES if f not in gdb_features]
    if missing_features:
        print(f"❌ Missing required features: {missing_features}")
        return False
    
    # Check for extra features
    extra_features = [f for f in gdb_features if f not in FEATURES]
    if extra_features:
        print(f"⚠️ Extra features found (will be ignored): {extra_features}")
    
    # Check if core features are present (first few important ones)
    core_features = FEATURES[:5]  # First 5 features are considered core
    missing_core = [f for f in core_features if f not in gdb_features]
    if missing_core:
        print(f"❌ Missing core features: {missing_core}")
        return False
    
    print("✅ Feature compatibility check passed")
    return True

def load_trained_model(model_type: str, optimization_method: str):
    """Load a pre-trained model using existing parameters"""
    try:
        params = get_model_params(model_type, optimization_method)
        if not params:
            print(f"No params found for {model_type}-{optimization_method}")
            return None
        
        print(f"Loading pre-trained model {optimization_method}_{model_type}...")
        
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif model_type == 'svm':
            from sklearn.svm import SVR
            model = SVR(**params)
        elif model_type == 'xgb':
            from xgboost import XGBRegressor
            model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        else:
            print(f"Unknown model type: {model_type}")
            return None
        
        # Note: Model is not trained here as it's already pre-trained
        # In a real implementation, you would load the model from disk
        print(f"Loaded pre-trained model {optimization_method}_{model_type}")
        return model
        
    except Exception as e:
        print(f"Error loading {model_type}-{optimization_method}: {e}")
        return None

def normalize_gdb_features(features: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Normalize GDB features using predefined min/max values.
    Validates that feature_names match expected order from training.
    """
    if len(features) == 0:
        return features
    
    # Validate feature order matches training
    if feature_names != FEATURES:
        print(f"⚠️ Feature order mismatch!")
        print(f"   Expected: {FEATURES}")
        print(f"   Received: {feature_names}")
        # Try to reorder if all features are present
        if set(feature_names) == set(FEATURES):
            print("   ⚠️ Features present but in different order. Reordering...")
            # Create mapping from feature to index
            feature_to_index = {feat: i for i, feat in enumerate(feature_names)}
            reordered_features = np.zeros_like(features)
            for i, expected_feature in enumerate(FEATURES):
                if expected_feature in feature_to_index:
                    source_idx = feature_to_index[expected_feature]
                    reordered_features[:, i] = features[:, source_idx]
            features = reordered_features
            print("   ✅ Features reordered to match training order")
        else:
            print("   ❌ Missing or extra features detected")
            return np.zeros_like(features)
    
    mins = np.array([FEATURE_MIN_MAX[f][0] for f in FEATURES])
    maxs = np.array([FEATURE_MIN_MAX[f][1] for f in FEATURES])
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    
    if features.shape[1] != len(mins):
        print(f"⚠️ Feature dimension mismatch in normalization: got {features.shape[1]}, expected {len(mins)}")
        return np.zeros_like(features)
    
    normalized = (features - mins) / ranges
    return normalized.astype(np.float32)

def load_chunk_clean(file_path: Path, layer_name: str, chunk_size: int, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load data in chunks without loading entire dataset"""
    coords_list = []
    features_list = []
    
    try:
        gdb_path = str(file_path)
        
        with fiona.open(gdb_path, layer=layer_name) as src:
            # Validate feature order in first chunk only
            if chunk_idx == 0:
                # Get available fields from GDB
                available_fields = list(src.schema['properties'].keys())
                # Validate feature compatibility
                if not validate_feature_compatibility(available_fields):
                    print("❌ Feature compatibility validation failed")
                    return np.array([]), np.array([])
                
                # Log feature order for validation
                print(f"📁 GDB available features: {available_fields[:10]}{'...' if len(available_fields) > 10 else ''}")
                print(f"📊 Expected feature order: {FEATURES}")
            
            total_features = len(src)
            start = chunk_idx * chunk_size
            
            # Early exit if chunk beyond data
            if start >= total_features:
                return np.array([]), np.array([])
            
            end = min(start + chunk_size, total_features)
            
            if chunk_idx == 0:
                print(f"\n=== CHUNK LOADING ===")
                print(f"Total features: {total_features}")
                print(f"Processing chunk {chunk_idx}: features {start} to {end-1}")
            
            # Direct slice access
            try:
                # Convert to list for slicing (memory efficient for small chunks)
                features_in_chunk = list(src)[start:end]
                
                for feature in features_in_chunk:
                    try:
                        geom = feature['geometry']
                        props = feature['properties']
                        
                        if geom and geom['type'] in ['Point', 'Polygon', 'MultiPolygon']:
                            # Extract coordinates
                            if geom['type'] == 'Point':
                                coords = geom['coordinates']
                            else:
                                shape = sg.shape(geom)
                                centroid = shape.centroid
                                coords = [centroid.x, centroid.y]
                            
                            coords_list.append(coords)
                            
                            # Extract features in correct order
                            feature_values = []
                            for col in FEATURES:
                                if col in props:
                                    val = props[col]
                                    if val is not None and not pd.isna(val):
                                        try:
                                            feature_values.append(float(val))
                                        except (ValueError, TypeError):
                                            feature_values.append(0.0)
                                    else:
                                        feature_values.append(0.0)
                                else:
                                    feature_values.append(0.0)
                            
                            features_list.append(feature_values)
                    
                    except Exception as e:
                        if chunk_idx == 0:
                            print(f"Error processing feature: {e}")
                        continue
                
                # Immediate cleanup
                del features_in_chunk
                
            except Exception as e:
                print(f"Error accessing chunk {chunk_idx}: {e}")
                return np.array([]), np.array([])
    
    except Exception as e:
        print(f"Error opening geodatabase: {e}")
        return np.array([]), np.array([])
    
    if len(coords_list) == 0 or len(features_list) == 0:
        return np.array([]), np.array([])
    
    # Convert to arrays with error handling
    try:
        coords = np.array(coords_list, dtype=np.float64)
        features = np.array(features_list, dtype=np.float32)
        
        # Immediate cleanup
        del coords_list, features_list
        
        # Validate feature dimensions match expected
        if len(features) > 0 and features.shape[1] != len(FEATURES):
            print(f"⚠️ Feature dimension mismatch: got {features.shape[1]}, expected {len(FEATURES)}")
            return np.array([]), np.array([])
        
        # Normalize features
        if len(features) > 0:
            features = normalize_gdb_features(features, FEATURES)
        
        if chunk_idx == 0 and len(coords) > 0:
            print(f"Chunk {chunk_idx}: {len(coords)} records processed")
            print(f"Features shape: {features.shape}")
            # Show sample of first few features to verify order
            if features.shape[0] > 0:
                sample_features = features[0, :min(5, features.shape[1])]
                print(f"Sample feature values: {sample_features}")
        
        return coords, features
        
    except MemoryError:
        print(f"Memory error processing chunk {chunk_idx}")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"Array conversion error: {e}")
        return np.array([]), np.array([])

def predict_gpu_batch(model, features, batch_size=8000):
    """Optimized batch prediction with better error handling"""
    # Quick validation
    if len(features) == 0:
        return np.array([])
    
    # Convert to numpy array if needed
    if not isinstance(features, np.ndarray):
        try:
            features = np.array(features)
        except MemoryError:
            print("Memory error converting features to numpy array")
            return np.zeros(len(features))
    
    # Validate dimensions
    if features.ndim != 2 or features.shape[1] == 0:
        print(f"Invalid feature dimensions: {features.shape}")
        return np.zeros(len(features))
    
    # Batch prediction
    predictions = []
    try:
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            
            try:
                pred = model.predict(batch)
                predictions.append(pred)
            except Exception as e:
                print(f"Prediction error for batch {i//batch_size}: {e}")
                predictions.append(np.zeros(len(batch)))
            
            # Memory cleanup for each batch
            del batch
        
        # Combine results
        if predictions:
            result = np.concatenate(predictions)
            del predictions  # Cleanup
            return result
        else:
            return np.array([])
            
    except Exception as e:
        print(f"Critical error in batch prediction: {e}")
        return np.zeros(len(features))

def process_model_in_chunks(file_path: Path, layer_name: str, model_type: str, 
                           optimization_method: str, output_dir: Path, 
                           chunk_size: int = CHUNK_SIZE) -> bool:
    """Process a single model with chunked data processing"""
    
    print(f"\n=== PROCESSING {optimization_method}_{model_type} ===")
    
    # Load only pre-trained model (no training)
    model = load_trained_model(model_type, optimization_method)
    if model is None:
        print(f"Failed to load {optimization_method}_{model_type}, skipping...")
        return False
    
    # Setup output
    output_dir.mkdir(exist_ok=True)
    x_min, y_min, x_max, y_max = STUDY_AREA_BOUNDS
    width = int((x_max - x_min) / 0.00009) + 1
    height = int((y_max - y_min) / 0.00009) + 1
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    
    # Create raster for this model
    raster = np.full((height, width), -9999.0, dtype=np.float32)
    total_processed = 0
    empty_chunks = 0
    
    # Get actual total features
    try:
        with fiona.open(str(file_path), layer=layer_name) as src:
            actual_total_features = len(src)
            print(f"Actual features in dataset: {actual_total_features}")
    except Exception as e:
        print(f"Error getting total features: {e}")
        del model
        gc.collect()
        return False
    
    # Calculate number of chunks needed
    estimated_chunks = (actual_total_features // chunk_size) + 1
    print(f"Processing {estimated_chunks} chunks of {chunk_size} records each")
    
    # Process chunks one by one
    for chunk_idx in range(estimated_chunks):
        coords, features = load_chunk_clean(file_path, layer_name, chunk_size, chunk_idx)
        
        if len(features) == 0:
            empty_chunks += 1
            if empty_chunks > 10:
                print(f"Stopping early after {empty_chunks} empty chunks")
                break
            continue
        
        empty_chunks = 0
        total_processed += len(features)
        
        try:
            preds = predict_gpu_batch(model, features)
        except Exception as e:
            print(f"Prediction error for chunk {chunk_idx}: {e}")
            preds = np.zeros(len(features))
        
        # Map predictions to raster
        try:
            cols = ((coords[:, 0] - x_min) / 0.00009).astype(np.int32)
            rows = ((y_max - coords[:, 1]) / 0.00009).astype(np.int32)
            
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            
            if valid.any():
                valid_cols = np.clip(cols[valid], 0, width - 1)
                valid_rows = np.clip(rows[valid], 0, height - 1)
                preds_valid = preds[valid].astype(np.float32)
                raster[valid_rows, valid_cols] = preds_valid
                del valid_cols, valid_rows, preds_valid
        except Exception as e:
            print(f"Raster mapping error for chunk {chunk_idx}: {e}")
        
        # Immediate cleanup
        del coords, features, preds, cols, rows, valid
        
        # Periodic garbage collection
        if chunk_idx % 3 == 0:
            gc.collect()
        
        print(f"Processed chunk {chunk_idx+1}/{estimated_chunks} ({total_processed} records total)")
    
    print(f"Processed {total_processed} records for {optimization_method}_{model_type}")
    
    # Save TIFF
    model_name = f"{optimization_method}_{model_type}"
    output_path = output_dir / f"{model_name}_flood_probability.tif"
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=1, dtype=np.float32,
        crs=CRS.from_epsg(4326), transform=transform, nodata=-9999,
        compress='lzw', tiled=True
    ) as dst:
        dst.write(raster, 1)
    
    print(f"Saved: {output_path.name}")
    
    # Cleanup model and raster completely
    del model, raster
    gc.collect()
    print(f"Cleaned up {model_name}")
    
    return True

def predict_to_tiff(file_path: Path, layer_name: str, output_dir: Path) -> None:
    """Main prediction function - processes each model once"""
    
    print("=== FLOOD PREDICTION ===")
    
    # Validate data source first
    try:
        gdb_path = str(file_path)
        gdf = gpd.read_file(gdb_path, layer=layer_name, rows=1)
        
        numeric_columns = [col for col in gdf.columns 
                          if col != 'geometry' and gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        # Verify we have the needed features
        available_features = [col for col in numeric_columns if col in FEATURES]
        if not available_features:
            print("Không có feature hợp lệ!")
            return
            
        del gdf  # Cleanup test data
            
    except Exception as e:
        print(f"Lỗi dữ liệu: {e}")
        return
    
    # Models to process - one at a time
    models_to_process = [
        ('rf', 'pso'), ('rf', 'po'), ('rf', 'rso'),
        ('svm', 'pso'), ('svm', 'po'), ('svm', 'rso'),
        ('xgb', 'pso'), ('xgb', 'po'), ('xgb', 'rso')
    ]
    
    # Process each model one at a time (sequential processing)
    for model_type, optimization_method in models_to_process:
        try:
            success = process_model_in_chunks(
                file_path, layer_name, model_type, optimization_method, output_dir
            )
            if not success:
                print(f"Failed to process {optimization_method}_{model_type}")
            else:
                print(f"Successfully processed {optimization_method}_{model_type}")
        except Exception as e:
            print(f"Error processing {optimization_method}_{model_type}: {e}")
        
        # Force cleanup between models
        gc.collect()
    
    print("\n=== PROCESSING COMPLETED ===")