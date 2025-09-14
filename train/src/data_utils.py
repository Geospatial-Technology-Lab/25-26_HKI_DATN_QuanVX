"""Simple, clean data utilities for flood prediction."""

import numpy as np
import geopandas as gpd
import joblib
from pathlib import Path
from typing import Tuple
from feature_config import FEATURES, FEATURE_MAPPING


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0,1]."""
    if len(features) == 0:
        return features
    fmin, fmax = features.min(axis=0), features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1
    return (features - fmin) / frange


def map_features(columns: list) -> dict:
    """Map columns to standard features."""
    mapped = {}
    
    print("=" * 80)
    print("📊 COLUMN MAPPING ANALYSIS")
    print("=" * 80)
    
    print(f"� ALL COLUMNS IN DATA ({len(columns)} total):")
    for i, col in enumerate(columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n🎯 REQUIRED FEATURES ({len(FEATURES)} total):")
    for i, feature in enumerate(FEATURES, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n🔗 FEATURE MAPPING CONFIGURATION:")
    for key, value in FEATURE_MAPPING.items():
        print(f"   '{key}' -> '{value}'")
    
    print(f"\n🔍 MAPPING PROCESS:")
    found_features = []
    missing_features = []
    
    for col in columns:
        if col in FEATURE_MAPPING and FEATURE_MAPPING[col] in FEATURES:
            mapped[FEATURE_MAPPING[col]] = col
            found_features.append(col)
            print(f"   ✅ '{col}' -> '{FEATURE_MAPPING[col]}' (MAPPED)")
        elif col in FEATURE_MAPPING:
            print(f"   ⚠️  '{col}' exists in mapping but target '{FEATURE_MAPPING[col]}' not in FEATURES")
        else:
            print(f"   ❌ '{col}' (NOT IN MAPPING)")
    
    # Check which features are missing
    for feature in FEATURES:
        if feature not in mapped:
            missing_features.append(feature)
    
    print(f"\n📈 SUMMARY:")
    print(f"   ✅ Successfully mapped: {len(mapped)}/{len(FEATURES)} features")
    print(f"   ✅ Found columns: {found_features}")
    
    if missing_features:
        print(f"   ❌ Missing features: {missing_features}")
        
        # Check for potential matches
        print(f"\n🔍 POTENTIAL MATCHES FOR MISSING FEATURES:")
        for missing in missing_features:
            similar_cols = [col for col in columns if missing.lower() in col.lower() or col.lower() in missing.lower()]
            if similar_cols:
                print(f"   - '{missing}' might match: {similar_cols}")
            else:
                print(f"   - '{missing}' no similar column found")
    
    print("=" * 80)
    return mapped


def debug_data_columns(file_path: Path, layer_name: str = None) -> None:
    """Debug function to check available columns in data."""
    try:
        print(f"🔍 Debugging data file: {file_path}")
        gdf = gpd.read_file(str(file_path), layer=layer_name, rows=5)
        
        print(f"📊 Total columns found: {len(gdf.columns)}")
        print(f"📋 All columns: {list(gdf.columns)}")
        
        # Check feature mapping
        features_map = map_features(gdf.columns)
        print(f"✅ Successfully mapped: {features_map}")
        
    except Exception as e:
        print(f"❌ Debug error: {e}")


def load_sample(file_path: Path, layer_name: str = None, n: int = 10) -> dict:
    """Load sample data."""
    try:
        gdf = gpd.read_file(str(file_path), layer=layer_name, rows=n)
        return {
            'gdf': gdf,
            'columns': list(gdf.columns),
            'features': map_features(gdf.columns),
            'size': len(gdf)
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def load_chunk(file_path: Path, layer_name: str, chunk_size: int, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load data chunk."""
    try:
        # Load full data (TODO: optimize for very large files)
        gdf = gpd.read_file(str(file_path), layer=layer_name)
        
        # Get chunk slice
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(gdf))
        if start >= len(gdf):
            return np.array([]), np.array([])
        
        chunk = gdf.iloc[start:end]
        features_map = map_features(chunk.columns)
        
        if not features_map:
            return np.array([]), np.array([])
        
        # Extract data
        coords = np.array([[g.x, g.y] for g in chunk.geometry])
        feature_cols = [features_map[f] for f in features_map.keys()]
        features = chunk[feature_cols].dropna().values
        
        return coords, normalize_features(features)
        
    except Exception as e:
        print(f"❌ Chunk error: {e}")
        return np.array([]), np.array([])


def load_models(model_dir: Path) -> dict:
    """Load all models."""
    models = {}
    for f in Path(model_dir).glob("*.joblib"):
        models[f.stem] = joblib.load(f)
    for f in Path(model_dir).glob("*.pkl"):
        models[f.stem] = joblib.load(f)
    return models


def get_row_count(file_path: Path, layer_name: str = None) -> int:
    """Get total rows."""
    try:
        gdf = gpd.read_file(str(file_path), layer=layer_name)
        return len(gdf)
    except:
        # Fallback estimate
        return int(file_path.stat().st_size / 1024 / 1024 * 5000)


# Legacy compatibility
load_sample_data = load_sample
load_all_models = load_models
get_total_row_count = get_row_count
load_and_process_geodata_chunk = load_chunk