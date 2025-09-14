import numpy as np
import geopandas as gpd
import fiona
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from feature_config import FEATURES, FEATURE_MAPPING


def normalize_features(features: np.ndarray) -> np.ndarray:

    if len(features) == 0:
        return features
        
    fmin, fmax = features.min(axis=0), features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1  # Avoid division by zero
    features_normalized = (features - fmin) / frange
    
    print(f"📊 Normalized features shape: {features_normalized.shape}")
    return features_normalized


def map_features_from_gdb(gdf_columns: list) -> dict:

    mapped_features = {}
    
    for col in gdf_columns:
        if col in FEATURE_MAPPING:
            standard_name = FEATURE_MAPPING[col]
            if standard_name in FEATURES:
                mapped_features[standard_name] = col
                print(f"✅ Mapped: {col} → {standard_name}")
    
    # Check for missing features
    missing_features = set(FEATURES) - set(mapped_features.keys())
    if missing_features:
        print(f"⚠️ Thiếu features: {missing_features}")
    
    available_features = list(mapped_features.keys())
    print(f"✅ Sử dụng {len(available_features)}/{len(FEATURES)} features")
    
    return mapped_features


def auto_detect_layer(file_path: Path) -> Optional[str]:

    if not str(file_path).endswith('.gpd'):
        return None
        
    try:
        layers = fiona.listlayers(str(file_path))
        return layers[0] if layers else None
    except Exception as e:
        print(f"⚠️ Could not detect layers: {e}")
        return None


def load_and_process_geodata(file_path: Path, layer_name: Optional[str] = None, 
                            start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                            count_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    try:
        # Auto-detect layer if needed
        if not layer_name:
            layer_name = auto_detect_layer(file_path)
        
        print(f"🔄 Loading từ layer: {layer_name}")
        
        # Load geodata
        if layer_name:
            gdf = gpd.read_file(str(file_path), layer=layer_name)
        else:
            gdf = gpd.read_file(str(file_path))
        
        # Handle chunking for large datasets
        if start_idx is not None and end_idx is not None:
            gdf = gdf.iloc[start_idx:end_idx]
        
        if count_only:
            coordinates = np.array([[geom.x, geom.y] for geom in gdf.geometry])
            return coordinates, np.array([])
        
        print(f"🔍 Columns trong file: {list(gdf.columns)}")
        
        # Map features
        mapped_features = map_features_from_gdb(gdf.columns)
        
        if not mapped_features:
            print("❌ No valid features found")
            return np.array([]), np.array([])
        
        # Filter data with original column names
        original_col_names = [mapped_features[feat] for feat in mapped_features.keys()]
        gdf_filtered = gdf[original_col_names + ['geometry']].dropna()
        
        # Extract coordinates and features
        coordinates = np.array([[geom.x, geom.y] for geom in gdf_filtered.geometry])
        features = gdf_filtered[original_col_names].values
        
        # Normalize features
        features_normalized = normalize_features(features)
        
        print(f"✅ Loaded: {len(coordinates):,} điểm, {len(mapped_features)} features")
        return coordinates, features_normalized
        
    except Exception as e:
        print(f"❌ Load error: {e}")
        return np.array([]), np.array([])


def load_all_models(model_dir: Path) -> Dict[str, Any]:

    models = {}
    model_dir = Path(model_dir)
    
    for model_file in list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl")):
        models[model_file.stem] = joblib.load(model_file)
    
    return models


def estimate_data_size(file_path: Path) -> int:

    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        return int(file_size_mb * 5000)  # rough estimate
    except:
        return 0