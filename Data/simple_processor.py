"""
Feature Processor - Tối giản
Xử lý 13 features cho điểm lưới
"""

import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

try:
    from simple_gpu import gpu
except ImportError:
    gpu = None

# Mapping features tự động
FEATURE_PATTERNS = {
    'dem': ['dem', 'elevation', 'height'],
    'slope': ['slope', 'gradient'],
    'aspect': ['aspect', 'direction'], 
    'curvature': ['curvature', 'curve'],
    'twi': ['twi', 'wetness'],
    'spi': ['spi', 'stream_power'],
    'ndvi': ['ndvi', 'vegetation'],
    'dist_river': ['river', 'stream', 'water'],
    'dist_road': ['road', 'highway'],
    'landuse': ['landuse', 'land_use', 'lulc'],
    'soil': ['soil', 'soil_type'],
    'precip': ['precip', 'rainfall', 'rain'],
    'flood_risk': ['flood', 'susceptibility', 'hazard']
}

def find_tiff_files(tiff_folder):
    """Tìm và map file TIFF với features"""
    folder = Path(tiff_folder)
    tiff_files = {}
    
    # Tìm tất cả file TIFF
    all_tiffs = []
    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        all_tiffs.extend(folder.glob(f'*{ext}'))
    
    print(f"Found {len(all_tiffs)} TIFF files")
    
    # Map với features
    for tiff in all_tiffs:
        filename = tiff.name.lower()
        
        for feature, patterns in FEATURE_PATTERNS.items():
            if any(pattern in filename for pattern in patterns):
                tiff_files[feature] = tiff
                print(f"Mapped {feature}: {tiff.name}")
                break
    
    missing = set(FEATURE_PATTERNS.keys()) - set(tiff_files.keys())
    if missing:
        print(f"Missing: {missing}")
    
    return tiff_files

def extract_values(tiff_path, points):
    """Trích xuất giá trị từ TIFF cho các điểm"""
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            transform = src.transform
            
            # GPU processing nếu có
            if gpu and gpu.available:
                gpu_data = gpu.to_gpu(data)
                values = _extract_gpu(gpu_data, points, transform)
                gpu.clear()
            else:
                values = _extract_cpu(data, points, transform)
            
            return values
            
    except Exception as e:
        print(f"Error extracting from {tiff_path}: {e}")
        return [np.nan] * len(points)

def _extract_gpu(gpu_data, points, transform):
    """GPU extraction"""
    try:
        values = []
        
        # Process in batches
        batch_size = 5000
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            # Convert to pixel coordinates
            x_coords = np.array([p.x for p in batch])
            y_coords = np.array([p.y for p in batch])
            
            cols = ((x_coords - transform.c) / transform.a).astype(int)
            rows = ((y_coords - transform.f) / transform.e).astype(int)
            
            # Check bounds
            valid = ((cols >= 0) & (cols < gpu_data.shape[1]) & 
                    (rows >= 0) & (rows < gpu_data.shape[0]))
            
            # Extract values
            for row, col, v in zip(rows, cols, valid):
                if v:
                    val = float(gpu.to_cpu(gpu_data[row, col]))
                    values.append(val)
                else:
                    values.append(np.nan)
        
        return values
        
    except Exception as e:
        print(f"GPU extraction failed: {e}")
        return _extract_cpu(gpu.to_cpu(gpu_data), points, transform)

def _extract_cpu(data, points, transform):
    """CPU extraction"""
    values = []
    
    for point in points:
        col = int((point.x - transform.c) / transform.a)
        row = int((point.y - transform.f) / transform.e)
        
        if (0 <= col < data.shape[1] and 0 <= row < data.shape[0]):
            values.append(data[row, col])
        else:
            values.append(np.nan)
    
    return values

def process_feature(feature, tiff_path, points):
    """Xử lý 1 feature"""
    try:
        print(f"Processing {feature}...")
        values = extract_values(tiff_path, points)
        print(f"Completed {feature}")
        return feature, values
    except Exception as e:
        print(f"Error in {feature}: {e}")
        return feature, None

def process_all_features(grid_shp, tiff_folder, max_workers=4):
    """
    Xử lý tất cả features
    
    Args:
        grid_shp: Đường dẫn shapefile lưới điểm
        tiff_folder: Thư mục chứa TIFF files
        max_workers: Số worker song song
    
    Returns:
        bool: Thành công hay không
    """
    
    # Load grid
    try:
        gdf = gpd.read_file(grid_shp)
        print(f"Loaded {len(gdf)} grid points")
    except Exception as e:
        print(f"Error loading grid: {e}")
        return False
    
    # Find TIFF files
    tiff_files = find_tiff_files(tiff_folder)
    if not tiff_files:
        print("No TIFF files found")
        return False
    
    # Get points
    points = gdf.geometry.tolist()
    
    # Process features in parallel
    print(f"Processing {len(tiff_files)} features with {max_workers} workers...")
    
    completed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_feature, feature, path, points): feature
            for feature, path in tiff_files.items()
        }
        
        for future in as_completed(futures):
            feature = futures[future]
            try:
                feat_name, values = future.result()
                if values is not None:
                    gdf[feat_name] = values
                    completed.append(feat_name)
            except Exception as e:
                print(f"Error in {feature}: {e}")
    
    print(f"Completed {len(completed)} features")
    
    # Save updated shapefile
    if completed:
        # Backup
        backup = Path(grid_shp).with_suffix('.backup.shp')
        if Path(grid_shp).exists():
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = Path(grid_shp).with_suffix(ext)
                dst = backup.with_suffix(ext)
                if src.exists():
                    shutil.copy2(src, dst)
        
        # Save
        gdf.to_file(grid_shp, driver='ESRI Shapefile')
        print(f"Updated: {grid_shp}")
        return True
    
    return False

def get_summary(grid_shp):
    """Lấy tóm tắt kết quả"""
    try:
        gdf = gpd.read_file(grid_shp)
        
        summary = {'total_points': len(gdf), 'features': {}}
        
        for feature in FEATURE_PATTERNS.keys():
            if feature in gdf.columns:
                values = gdf[feature]
                summary['features'][feature] = {
                    'valid': values.notna().sum(),
                    'missing': values.isna().sum(),
                    'mean': values.mean() if values.notna().any() else np.nan,
                    'min': values.min() if values.notna().any() else np.nan,
                    'max': values.max() if values.notna().any() else np.nan
                }
        
        return summary
        
    except Exception as e:
        print(f"Error getting summary: {e}")
        return None

# Main workflow
def complete_workflow(ref_tiff, tiff_folder, output_folder, grid_size=(100, 100), max_workers=4):
    """
    Workflow hoàn chỉnh: tạo grid + xử lý features
    
    Args:
        ref_tiff: File TIFF tham chiếu để tạo grid
        tiff_folder: Thư mục chứa 13 file TIFF features
        output_folder: Thư mục đầu ra
        grid_size: Kích thước ô lưới (m)
        max_workers: Số worker
    
    Returns:
        bool: Thành công
    """
    from simple_grid import create_point_grid
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    grid_shp = output_path / "flood_grid.shp"
    
    print("=== CREATING GRID ===")
    grid = create_point_grid(ref_tiff, grid_shp, grid_size)
    if grid is None:
        return False
    
    print("\n=== PROCESSING FEATURES ===")
    success = process_all_features(grid_shp, tiff_folder, max_workers)
    
    if success:
        print("\n=== SUMMARY ===")
        summary = get_summary(grid_shp)
        if summary:
            print(f"Total points: {summary['total_points']}")
            for feat, stats in summary['features'].items():
                print(f"{feat}: {stats['valid']} valid values")
    
    return success

if __name__ == "__main__":
    # Test workflow
    ref_tiff = "reference.tif"
    tiff_folder = "tiff_files/"
    output_folder = "output/"
    
    success = complete_workflow(
        ref_tiff=ref_tiff,
        tiff_folder=tiff_folder, 
        output_folder=output_folder,
        grid_size=(100, 100),
        max_workers=6
    )
    
    print("SUCCESS!" if success else "FAILED!")
