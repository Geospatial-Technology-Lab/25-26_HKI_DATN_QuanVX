"""
Grid Creator - Tối giản
Tạo điểm lưới từ TIFF chỉ cho ô có dữ liệu
"""

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path

try:
    from simple_gpu import gpu
except ImportError:
    gpu = None

def create_point_grid(tiff_path, output_shp, grid_size=None):
    """
    Tạo lưới điểm từ TIFF
    
    Args:
        tiff_path: Đường dẫn file TIFF
        output_shp: Đường dẫn file shapefile đầu ra  
        grid_size: Kích thước ô lưới (width, height)
    
    Returns:
        GeoDataFrame: Lưới điểm
    """
    
    # Đọc raster
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        crs = src.crs
        data = src.read(1)
        nodata = src.nodata
        
        # Kích thước mặc định
        if grid_size is None:
            grid_size = (abs(src.transform[0]), abs(src.transform[4]))
    
    print(f"Creating grid from {tiff_path}")
    print(f"Grid size: {grid_size}")
    
    # Tính số ô lưới
    n_cols = int(np.ceil((bounds.right - bounds.left) / grid_size[0]))
    n_rows = int(np.ceil((bounds.top - bounds.bottom) / grid_size[1]))
    
    # Tạo tọa độ
    x_coords = np.linspace(bounds.left, bounds.left + n_cols * grid_size[0], n_cols + 1)
    y_coords = np.linspace(bounds.bottom, bounds.bottom + n_rows * grid_size[1], n_rows + 1)
    
    # Tạo điểm chỉ cho ô có dữ liệu
    points = []
    point_data = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Tọa độ trung tâm ô
            x = (x_coords[col] + x_coords[col + 1]) / 2
            y = (y_coords[row] + y_coords[row + 1]) / 2
            
            # Kiểm tra pixel có dữ liệu
            px_col = min(max(int(((x - bounds.left) / (bounds.right - bounds.left)) * 
                                (data.shape[1] - 1)), 0), data.shape[1] - 1)
            px_row = min(max(int(((bounds.top - y) / (bounds.top - bounds.bottom)) * 
                                (data.shape[0] - 1)), 0), data.shape[0] - 1)
            
            pixel_val = data[px_row, px_col]
            
            # Chỉ tạo điểm nếu có dữ liệu hợp lệ
            if nodata is None or (not np.isnan(pixel_val) and pixel_val != nodata):
                points.append(Point(x, y))
                point_data.append({
                    'id': len(points) - 1,
                    'row': row,
                    'col': col,
                    'x': x,
                    'y': y
                })
    
    print(f"Created {len(points)} points (out of {n_cols * n_rows} total cells)")
    
    # Tạo GeoDataFrame
    df = pd.DataFrame(point_data)
    df['geometry'] = points
    
    # Thêm cột features
    features = [
        'dem', 'slope', 'aspect', 'curvature', 'twi', 'spi', 'ndvi',
        'dist_river', 'dist_road', 'landuse', 'soil', 'precip', 'flood_risk'
    ]
    
    for feature in features:
        df[feature] = np.nan
    
    gdf = gpd.GeoDataFrame(df, crs=crs)
    
    # Lưu shapefile
    output_path = Path(output_shp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Xóa file cũ
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        old_file = output_path.with_suffix(ext)
        if old_file.exists():
            old_file.unlink()
    
    gdf.to_file(output_shp, driver='ESRI Shapefile')
    print(f"Saved: {output_shp}")
    
    return gdf

if __name__ == "__main__":
    # Test
    tiff = "test.tif"
    output = "grid_points.shp"
    grid = create_point_grid(tiff, output, (100, 100))
    print(f"Grid created with {len(grid)} points")
