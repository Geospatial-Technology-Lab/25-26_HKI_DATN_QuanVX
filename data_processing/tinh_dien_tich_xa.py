# -*- coding: utf-8 -*-
"""
Script tính diện tích ngưỡng lũ lụt theo từng xã
Phân tích file shapefile xã và ảnh tiff ngưỡng lũ lụt
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# Cấu hình font tiếng Việt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def read_shapefile(file_path):
    """Đọc file shapefile"""
    try:
        gdf = gpd.read_file(file_path)
        print(f"✓ Đọc shapefile thành công: {len(gdf)} xã")
        return gdf
    except Exception as e:
        print(f"✗ Lỗi khi đọc shapefile: {e}")
        return None

def calculate_area_by_threshold(shapefile_path, tiff_path):
    """
    Tính diện tích theo từng ngưỡng cho mỗi xã
    
    Args:
        shapefile_path: Đường dẫn file shapefile
        tiff_path: Đường dẫn file tiff
    
    Returns:
        DataFrame: Bảng diện tích theo xã và ngưỡng
    """
    # Đọc shapefile
    gdf = read_shapefile(shapefile_path)
    if gdf is None:
        return None
    
    # Đọc file tiff
    print(f"Đang đọc file TIFF: {tiff_path}")
    
    results = []
    
    with rasterio.open(tiff_path) as src:
        print(f"✓ Đọc TIFF thành công")
        print(f"  - Kích thước: {src.width} x {src.height}")
        print(f"  - CRS: {src.crs}")
        
        # Đảm bảo cùng hệ tọa độ với raster
        if gdf.crs != src.crs:
            print(f"Chuyển đổi CRS từ {gdf.crs} sang {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Tính diện tích pixel
        # Với EPSG:4326, transform là độ (degrees)
        # Chuyển đổi độ sang mét tại vĩ độ trung bình
        # 1 độ kinh tuyến ≈ 111,320 mét
        # 1 độ vĩ tuyến ≈ 111,320 * cos(vĩ độ) mét
        
        # Lấy vĩ độ trung bình của khu vực
        bounds = src.bounds
        lat_center = (bounds.bottom + bounds.top) / 2
        
        # Độ phân giải (degrees)
        pixel_width_deg = abs(src.transform[0])
        pixel_height_deg = abs(src.transform[4])
        
        # Chuyển sang mét
        meters_per_deg_lat = 111320  # mét/độ vĩ tuyến
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))  # mét/độ kinh tuyến
        
        pixel_width_m = pixel_width_deg * meters_per_deg_lon
        pixel_height_m = pixel_height_deg * meters_per_deg_lat
        pixel_area = pixel_width_m * pixel_height_m
        
        print(f"Diện tích pixel: {pixel_area:.2f} m² ({pixel_area/1_000_000:.6f} km²)")
        
        print(f"\nĐang xử lý {len(gdf)} xã...")
        
        for idx, row in gdf.iterrows():
            ten_xa = row['ten_xa']
            ma_xa = row['ma_xa']
            
            try:
                # Cắt raster theo từng xã
                geom = [row['geometry']]
                out_image, out_transform = mask(src, geom, crop=True, filled=False)
                
                # Lấy dữ liệu band đầu tiên
                data = out_image[0]
                
                # Xử lý masked array - chỉ lấy các pixel không phải NoData
                if np.ma.is_masked(data):
                    # Lấy data không bị mask
                    valid_data = data[~data.mask]
                else:
                    # Không có mask, loại bỏ giá trị 0 (NoData)
                    valid_data = data[data != 0]
                
                # Tính diện tích cho từng ngưỡng (1-5)
                area_dict = {
                    'ma_xa': ma_xa,
                    'ten_xa': ten_xa
                }
                
                for threshold in range(1, 6):
                    # Đếm số pixel có giá trị = threshold
                    count = np.sum(valid_data == threshold)
                    # Tính diện tích (m²) và chuyển sang km²
                    area_km2 = (count * pixel_area) / 1_000_000
                    area_dict[f'nguong_{threshold}'] = area_km2
                
                results.append(area_dict)
                
                if (idx + 1) % 20 == 0:
                    print(f"  Đã xử lý {idx + 1}/{len(gdf)} xã")
                    
            except Exception as e:
                print(f"  ⚠ Lỗi xử lý xã {ten_xa}: {e}")
                continue
    
    print(f"✓ Hoàn thành xử lý {len(results)} xã")
    
    # Tạo DataFrame
    df = pd.DataFrame(results)
    
    # Chuyển đổi các cột diện tích sang kiểu số
    for i in range(1, 6):
        col_name = f'nguong_{i}'
        df[col_name] = df[col_name].astype(float)
    
    # Tính tổng ngưỡng cao (4) và rất cao (5)
    df['tong_cao_ratcao'] = df['nguong_4'].astype(float) + df['nguong_5'].astype(float)
    
    return df

def get_top_10_communes(df):
    """Lấy 10 xã có tổng diện tích ngưỡng cao và rất cao lớn nhất"""
    top_10 = df.nlargest(10, 'tong_cao_ratcao')
    
    print("\n" + "="*80)
    print("TOP 10 XÃ CÓ TỔNG DIỆN TÍCH NGƯỠNG CAO VÀ RẤT CAO LỚN NHẤT")
    print("="*80)
    print(f"{'STT':<5}{'Tên xã':<25}{'Mã xã':<10}{'Ngưỡng Cao (km²)':<20}{'Ngưỡng Rất Cao (km²)':<22}{'Tổng (km²)':<15}")
    print("-"*80)
    
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:<5}{row['ten_xa']:<25}{row['ma_xa']:<10}{row['nguong_4']:<20.2f}{row['nguong_5']:<22.2f}{row['tong_cao_ratcao']:<15.2f}")
    
    return top_10

def plot_threshold_lines(top_10_df, output_path=None):
    """
    Vẽ biểu đồ đường cho 10 xã
    
    Args:
        top_10_df: DataFrame của 10 xã hàng đầu
        output_path: Đường dẫn lưu hình ảnh
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Các ngưỡng
    thresholds = [1, 2, 3, 4, 5]
    threshold_labels = ['Rất thấp', 'Thấp', 'Trung bình', 'Cao', 'Rất cao']
    
    # Màu sắc cho 10 xã
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
              '#e67e22', '#1abc9c', '#34495e', '#c0392b', '#16a085']
    
    # Vẽ từng xã
    for i, (idx, row) in enumerate(top_10_df.iterrows()):
        areas = [row[f'nguong_{t}'] for t in thresholds]
        ax.plot(thresholds, areas, marker='o', linewidth=2.5, 
                markersize=8, color=colors[i], label=row['ten_xa'], alpha=0.8)
    
    # Cấu hình trục
    ax.set_xlabel('Mức độ nhạy cảm', fontsize=18, fontweight='bold')
    ax.set_ylabel('Diện tích (km²)', fontsize=18, fontweight='bold')
    
    # Thiết lập trục x
    ax.set_xticks(thresholds)
    ax.set_xticklabels(threshold_labels, fontsize=14)
    
    # Thiết lập lưới
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Chú giải ở góc trên bên trái
    ax.legend(loc='upper left', fontsize=14, frameon=True, shadow=True, 
              fancybox=True, framealpha=0.95)
    
    # Định dạng trục y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Đã lưu biểu đồ tại: {output_path}")
    
    plt.show()
    print("✓ Hiển thị biểu đồ")

def plot_specific_communes(df, output_path=None):

    # Danh sách tên xã cần vẽ
    target_communes = [
        'An Nhơn',
        'An Nhơn Đông',
        'An Nhơn Tây',
        'An Nhơn Nam',
        'An Nhơn Bắc',
        'Bình Định',
        'Cát Tiến',
        'Ngô Mây',
        'Quy Nhơn',
        'Quy Nhơn Bắc',
        'Quy Nhơn Đông',
        'Tuy Phước',
        'Tuy Phước Bắc',
        'Tuy Phước Đông',
        'Tuy Phước Tây',
        'Xuân An'
    ]
    
    # Lọc dữ liệu các xã cần vẽ
    filtered_df = df[df['ten_xa'].isin(target_communes)].copy()
    
    if len(filtered_df) == 0:
        print("⚠ Không tìm thấy xã nào trong danh sách!")
        print("Các xã có trong dữ liệu:")
        print(df['ten_xa'].unique())
        return
    
    # Sắp xếp các xã theo diện tích ngưỡng 5 (rất cao) từ lớn đến nhỏ
    filtered_df = filtered_df.sort_values('nguong_5', ascending=False)
    
    print(f"\nĐang vẽ biểu đồ cho {len(filtered_df)} xã (sắp xếp theo ngưỡng rất cao):")
    for xa, area in zip(filtered_df['ten_xa'].values, filtered_df['nguong_5'].values):
        print(f"  - {xa}: {area:.2f} km²")
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Các ngưỡng
    thresholds = [1, 2, 3, 4, 5]
    threshold_labels = ['Rất thấp', 'Thấp', 'Trung bình', 'Cao', 'Rất cao']
    
    # Màu sắc cho các xã (16 xã cần 16 màu khác nhau)
    colors = [
        '#e74c3c',  # Đỏ tươi
        '#3498db',  # Xanh dương sáng
        '#2ecc71',  # Xanh lá cây
        '#f39c12',  # Cam vàng
        "#bb82d1",  # Tím
        '#1abc9c',  # Xanh ngọc lam
        '#e67e22',  # Cam đậm
        '#34495e',  # Xám xanh đậm
        '#16a085',  # Xanh lục bảo
        '#c0392b',  # Đỏ thẫm
        '#d35400',  # Cam cháy
        '#27ae60',  # Xanh lá đậm
        "#8a198a",  # Tím đậm
        '#2c3e50',  # Xanh đen
        '#f1c40f',  # Vàng
        '#95a5a6'   # Xám bạc
    ]
    
    # Vẽ từng xã
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        areas = [row[f'nguong_{t}'] for t in thresholds]
        color = colors[i % len(colors)]
        ax.plot(thresholds, areas, marker='o', linewidth=2.5, 
                markersize=8, color=color, label=row['ten_xa'], alpha=0.85)
    
    # Cấu hình trục
    ax.set_xlabel('Mức độ nhạy cảm', fontsize=18, fontweight='bold')
    ax.set_ylabel('Diện tích (km²)', fontsize=18, fontweight='bold')

    # Thiết lập trục x
    ax.set_xticks(thresholds)
    ax.set_xticklabels(threshold_labels, fontsize=14)
    
    # Thiết lập lưới
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Chú giải - chia thành 2 cột để không bị dài
    ax.legend(loc='upper left', fontsize=13, frameon=True, shadow=True, 
              fancybox=True, framealpha=0.95, ncol=2)
    
    # Định dạng trục y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Đã lưu biểu đồ tại: {output_path}")
    
    plt.show()
    print("✓ Hiển thị biểu đồ các xã cụ thể")

def export_results(df, output_path):
    """Xuất kết quả ra file CSV"""
    df_sorted = df.sort_values('tong_cao_ratcao', ascending=False)
    df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Đã xuất kết quả ra file: {output_path}")

def main():
    """Hàm chính"""
    print("="*80)
    print("PHÂN TÍCH DIỆN TÍCH NGƯỠNG LŨ LỤT THEO XÃ")
    print("="*80)
    
    # Đường dẫn file
    shapefile_path = r"C:\Users\Admin\Desktop\GL\xa.shp"
    tiff_path = r"D:\prj\results\map\threshold\xgb\flood_susceptibility_po_XGB.tif"
    
    print(f"\nShapefile: {shapefile_path}")
    print(f"TIFF file: {tiff_path}")
    
    # Tính diện tích
    df_results = calculate_area_by_threshold(shapefile_path, tiff_path)
    
    if df_results is not None:
        # Lấy top 10 xã
        top_10 = get_top_10_communes(df_results)
        
        # === VẼ BIỂU ĐỒ TOP 10 XÃ CÓ NGƯỠNG CAO VÀ RẤT CAO CAO NHẤT ===
        # Comment đoạn này nếu không cần vẽ biểu đồ top 10
        # output_image = r"C:\Users\Admin\Desktop\GL\top10_xa_bieu_do.png"
        # plot_threshold_lines(top_10, output_image)
        
        # === VẼ BIỂU ĐỒ CÁC XÃ CỤ THỂ ===
        # Vẽ biểu đồ cho Tuy Phước, Quy Nhơn và các hướng
        output_image_specific = r"C:\Users\Admin\Desktop\GL\xa_cu_the_bieu_do.png"
        plot_specific_communes(df_results, output_image_specific)
        
        # Xuất kết quả toàn bộ
        output_csv = r"C:\Users\Admin\Desktop\GL\dien_tich_nguong_theo_xa.csv"
        export_results(df_results, output_csv)
        
        print("\n" + "="*80)
        print("HOÀN THÀNH!")
        print("="*80)

if __name__ == "__main__":
    main()
