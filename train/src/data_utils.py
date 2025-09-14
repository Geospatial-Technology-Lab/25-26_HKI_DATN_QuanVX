"""Simple, clean data utilities for flood prediction."""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from pathlib import Path
from typing import Tuple, Iterator
from tqdm import tqdm
from feature_config import FEATURES, FEATURE_MAPPING

# GPU support detection
try:
    import cudf
    import cupy as cp
    CUDA_AVAILABLE = True
    print("✅ GPU support available (cuDF/cuPy)")
except ImportError:
    CUDA_AVAILABLE = False
    print("ℹ️  Using CPU (pandas/numpy)")


def process_large_dataset(file_path: Path, output_path: Path = None, 
                         chunk_size: int = 10_000_000, use_gpu: bool = True) -> None:
    """Process large dataset with chunking, dropna, and min-max normalization.
    
    Args:
        file_path: Input CSV file path
        output_path: Output file path (auto-generated if None)
        chunk_size: Number of rows per chunk (default: 10 million)
        use_gpu: Use GPU if available
    """
    if output_path is None:
        output_path = file_path.parent / f"{file_path.stem}_processed{file_path.suffix}"
    
    use_gpu = use_gpu and CUDA_AVAILABLE
    
    print(f"🚀 Processing large dataset...")
    print(f"📥 Input: {file_path}")
    print(f"📤 Output: {output_path}")
    print(f"🔧 GPU enabled: {use_gpu}")
    print(f"📦 Chunk size: {chunk_size:,} rows")
    print("-" * 80)
    
    processed_chunks = []
    total_processed_rows = 0
    
    try:
        # Read file in chunks
        chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
        
        for chunk_idx, chunk in enumerate(tqdm(chunk_reader, desc="Processing chunks", unit="chunk")):
            print(f"\n📦 Chunk {chunk_idx + 1}:")
            print(f"  📊 Raw data: {len(chunk):,} rows, {len(chunk.columns)} columns")
            
            # Step 1: Remove missing data with dropna
            initial_rows = len(chunk)
            cleaned_chunk = chunk.dropna()
            final_rows = len(cleaned_chunk)
            dropped_rows = initial_rows - final_rows
            
            if dropped_rows > 0:
                print(f"  🧹 Dropped {dropped_rows:,} rows with missing data ({dropped_rows/initial_rows*100:.1f}%)")
            
            if len(cleaned_chunk) == 0:
                print("  ⚠️  No data left after cleaning, skipping chunk")
                continue
            
            # Step 2: Min-max normalization (x - x_min) / (x_max - x_min)
            numeric_cols = cleaned_chunk.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                print(f"  📏 Normalizing {len(numeric_cols)} numeric columns...")
                
                if use_gpu:
                    # GPU processing with cuDF
                    gpu_df = cudf.from_pandas(cleaned_chunk)
                    
                    for col in numeric_cols:
                        col_data = gpu_df[col]
                        col_min = col_data.min()
                        col_max = col_data.max()
                        
                        if col_max != col_min:
                            gpu_df[col] = (col_data - col_min) / (col_max - col_min)
                        else:
                            gpu_df[col] = 0.0
                    
                    normalized_chunk = gpu_df.to_pandas()
                else:
                    # CPU processing with pandas
                    normalized_chunk = cleaned_chunk.copy()
                    
                    for col in numeric_cols:
                        col_data = normalized_chunk[col]
                        col_min = col_data.min()
                        col_max = col_data.max()
                        
                        if col_max != col_min:
                            normalized_chunk[col] = (col_data - col_min) / (col_max - col_min)
                        else:
                            normalized_chunk[col] = 0.0
            else:
                normalized_chunk = cleaned_chunk
            
            print(f"  ✅ Processed: {len(normalized_chunk):,} rows")
            
            # Save chunk to avoid memory issues
            chunk_output = output_path.parent / f"{output_path.stem}_chunk_{chunk_idx + 1}{output_path.suffix}"
            normalized_chunk.to_csv(chunk_output, index=False)
            processed_chunks.append(chunk_output)
            
            total_processed_rows += len(normalized_chunk)
            print(f"  💾 Saved chunk to: {chunk_output.name}")
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        raise
    
    # Combine all chunks into final file
    if processed_chunks:
        print(f"\n🔗 Combining {len(processed_chunks)} chunks...")
        
        combined_data = []
        for chunk_file in tqdm(processed_chunks, desc="Combining chunks"):
            chunk_data = pd.read_csv(chunk_file)
            combined_data.append(chunk_data)
        
        final_data = pd.concat(combined_data, ignore_index=True)
        final_data.to_csv(output_path, index=False)
        
        # Clean up temporary chunk files
        for chunk_file in processed_chunks:
            chunk_file.unlink()
        
        print(f"✅ Processing complete!")
        print(f"📊 Final result: {len(final_data):,} rows, {len(final_data.columns)} columns")
        print(f"💾 Saved to: {output_path}")
    else:
        print("❌ No data to process")


def read_csv_chunks(file_path: Path, chunk_size: int = 10_000_000) -> Iterator[pd.DataFrame]:
    """Read CSV file in chunks with progress tracking."""
    print(f"📖 Reading file in chunks of {chunk_size:,} rows...")
    
    total_size = os.path.getsize(file_path)
    print(f"📊 File size: {total_size / (1024**3):.2f} GB")
    
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
    
    for chunk_idx, chunk in enumerate(chunk_reader):
        print(f"  Processing chunk {chunk_idx + 1}: {len(chunk):,} rows")
        yield chunk


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