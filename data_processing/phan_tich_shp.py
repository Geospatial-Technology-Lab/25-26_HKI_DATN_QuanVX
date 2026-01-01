"""
Script phân tích file shapefile
Đọc và phân tích thông tin từ file xa.shp
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def read_shapefile(file_path):
    """
    Đọc file shapefile
    
    Args:
        file_path: Đường dẫn đến file .shp
    
    Returns:
        GeoDataFrame: Dữ liệu shapefile
    """
    try:
        gdf = gpd.read_file(file_path)
        print(f"✓ Đọc file thành công: {file_path}")
        return gdf
    except Exception as e:
        print(f"✗ Lỗi khi đọc file: {e}")
        return None

def analyze_basic_info(gdf):
    """
    Phân tích thông tin cơ bản của shapefile
    
    Args:
        gdf: GeoDataFrame
    """
    print("\n" + "="*60)
    print("THÔNG TIN Cơ BẢN")
    print("="*60)
    
    print(f"\n1. Số lượng đối tượng (features): {len(gdf)}")
    print(f"2. Số lượng thuộc tính (columns): {len(gdf.columns)}")
    
    print(f"\n3. Hệ tọa độ (CRS): {gdf.crs}")
    
    print(f"\n4. Kiểu hình học:")
    geom_types = gdf.geometry.geom_type.value_counts()
    for geom_type, count in geom_types.items():
        print(f"   - {geom_type}: {count}")
    
    print(f"\n5. Các thuộc tính:")
    for i, col in enumerate(gdf.columns, 1):
        dtype = gdf[col].dtype
        print(f"   {i}. {col} ({dtype})")

def analyze_geometry(gdf):
    """
    Phân tích thông tin hình học
    
    Args:
        gdf: GeoDataFrame
    """
    print("\n" + "="*60)
    print("THÔNG TIN HÌNH HỌC")
    print("="*60)
    
    # Tính diện tích (nếu là polygon)
    if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
        gdf['area'] = gdf.geometry.area
        print(f"\n1. Diện tích:")
        print(f"   - Tổng diện tích: {gdf['area'].sum():.2f}")
        print(f"   - Diện tích trung bình: {gdf['area'].mean():.2f}")
        print(f"   - Diện tích lớn nhất: {gdf['area'].max():.2f}")
        print(f"   - Diện tích nhỏ nhất: {gdf['area'].min():.2f}")
    
    # Tính độ dài chu vi
    gdf['perimeter'] = gdf.geometry.length
    print(f"\n2. Chu vi/Độ dài:")
    print(f"   - Tổng chu vi: {gdf['perimeter'].sum():.2f}")
    print(f"   - Chu vi trung bình: {gdf['perimeter'].mean():.2f}")
    
    # Phạm vi không gian (bounding box)
    bounds = gdf.total_bounds
    print(f"\n3. Phạm vi không gian (Bounding Box):")
    print(f"   - Min X: {bounds[0]:.6f}")
    print(f"   - Min Y: {bounds[1]:.6f}")
    print(f"   - Max X: {bounds[2]:.6f}")
    print(f"   - Max Y: {bounds[3]:.6f}")

def analyze_attributes(gdf):
    """
    Phân tích thông tin thuộc tính
    
    Args:
        gdf: GeoDataFrame
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH THUỘC TÍNH")
    print("="*60)
    
    # Loại bỏ cột geometry để phân tích
    df = gdf.drop(columns=['geometry'])
    
    if len(df.columns) > 0:
        for col in df.columns:
            print(f"\n--- Thuộc tính: {col} ---")
            
            if df[col].dtype in ['int64', 'float64']:
                # Phân tích thuộc tính số
                print(f"Kiểu: Số")
                print(f"  - Số lượng giá trị: {df[col].count()}")
                print(f"  - Giá trị null: {df[col].isna().sum()}")
                print(f"  - Trung bình: {df[col].mean():.2f}")
                print(f"  - Độ lệch chuẩn: {df[col].std():.2f}")
                print(f"  - Min: {df[col].min()}")
                print(f"  - Max: {df[col].max()}")
            else:
                # Phân tích thuộc tính phân loại
                print(f"Kiểu: Phân loại/Text")
                print(f"  - Số lượng giá trị: {df[col].count()}")
                print(f"  - Giá trị null: {df[col].isna().sum()}")
                print(f"  - Số lượng giá trị duy nhất: {df[col].nunique()}")
                
                # Hiển thị các giá trị phổ biến nhất
                value_counts = df[col].value_counts().head(5)
                if len(value_counts) > 0:
                    print(f"  - Top 5 giá trị phổ biến:")
                    for val, count in value_counts.items():
                        print(f"    + {val}: {count}")



def main():
    """
    Hàm chính thực hiện phân tích
    """
    # Đường dẫn file shapefile
    shapefile_path = r"C:\Users\Admin\Desktop\GL\xa.shp"
    
    print("="*60)
    print("PHÂN TÍCH FILE SHAPEFILE")
    print("="*60)
    print(f"File: {shapefile_path}")
    
    # Đọc file shapefile
    gdf = read_shapefile(shapefile_path)
    
    if gdf is not None:
        # Phân tích thông tin cơ bản
        analyze_basic_info(gdf)
        
        # Phân tích thông tin hình học
        analyze_geometry(gdf)
        
        # Phân tích thuộc tính
        analyze_attributes(gdf)
        
        # Hiển thị 5 dòng đầu tiên
        print("\n" + "="*60)
        print("DỮ LIỆU MẪU (5 dòng đầu)")
        print("="*60)
        print(gdf.head())
        
        
        print("\n" + "="*60)
        print("HOÀN THÀNH PHÂN TÍCH!")
        print("="*60)

if __name__ == "__main__":
    main()
