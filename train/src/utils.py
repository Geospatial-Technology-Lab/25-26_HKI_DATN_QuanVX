import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import os
from config import OUTPUT_DIR, PIXEL_SIZE, CRS, TIFF_COMPRESS, TIFF_DTYPE, NODATA_VALUE
import warnings
warnings.filterwarnings('ignore')

def classify_flood_risk(probability):
    """
    Phân loại xác suất lũ lụt thành 5 ngưỡng rủi ro
    
    Args:
        probability: Xác suất lũ lụt (0-1)
    
    Returns:
        int: Mã rủi ro (1-5)
            1: Rủi ro rất thấp (0-0.2)
            2: Rủi ro thấp (0.2-0.4)
            3: Rủi ro trung bình (0.4-0.6)
            4: Rủi ro cao (0.6-0.8)
            5: Rủi ro rất cao (>0.8)
    """
    if probability < 0.2:
        return 1  # Rủi ro rất thấp
    elif probability < 0.4:
        return 2  # Rủi ro thấp
    elif probability < 0.6:
        return 3  # Rủi ro trung bình
    elif probability < 0.8:
        return 4  # Rủi ro cao
    else:
        return 5  # Rủi ro rất cao

def create_single_tiff(df, x_col, y_col, value_col, output_path, 
                      pixel_size=None, crs=None, compress=None, is_risk_map=False):

    if pixel_size is None:
        pixel_size = PIXEL_SIZE
    if crs is None:
        crs = CRS
    if compress is None:
        compress = TIFF_COMPRESS
    
    # Tạo thư mục output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lấy bounds (tọa độ địa lý)
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    print(f"   📐 Bounds: Lon({x_min:.4f}-{x_max:.4f}°), Lat({y_min:.4f}-{y_max:.4f}°)")
    
    # Kích thước raster
    width = int((x_max - x_min) / pixel_size) + 1
    height = int((y_max - y_min) / pixel_size) + 1
    
    print(f"   🖼️ Raster: {width}x{height} pixels ({pixel_size}°/pixel ≈ {pixel_size*111111:.0f}m)")
    
    if width * height > 50000000:  # 50M pixels
        print(f"   ⚠️ Raster rất lớn ({width*height:,} pixels), có thể chậm...")
    
    # Transform cho WGS84
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    
    # Tạo raster trống
    if is_risk_map:
        raster = np.zeros((height, width), dtype=np.uint8)  # Risk map dùng integer
        dtype = 'uint8'
        nodata_value = 0
    else:
        raster = np.full((height, width), NODATA_VALUE, dtype=np.float32)  # Probability map
        dtype = TIFF_DTYPE
        nodata_value = NODATA_VALUE
    
    # Điền giá trị từ DataFrame
    filled_pixels = 0
    for _, row in df.iterrows():
        if pd.isna(row[value_col]):
            continue
            
        col = int((row[x_col] - x_min) / pixel_size)
        row_idx = int((y_max - row[y_col]) / pixel_size)
        
        if 0 <= col < width and 0 <= row_idx < height:
            if is_risk_map:
                raster[row_idx, col] = classify_flood_risk(row[value_col])
            else:
                raster[row_idx, col] = row[value_col]
            filled_pixels += 1
    
    print(f"   💾 Điền {filled_pixels}/{len(df)} điểm vào raster")
    
    # Lưu TIFF
    try:
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height, width=width,
            count=1, dtype=dtype,
            crs=crs,
            transform=transform,
            compress=compress,
            nodata=nodata_value
        ) as dst:
            dst.write(raster, 1)
        
        print(f"✅ {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi lưu {os.path.basename(output_path)}: {e}")
        return False

def create_all_tiffs(df, output_dir=None):

    if output_dir is None:
        output_dir = OUTPUT_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm các cột probability
    prob_cols = [col for col in df.columns if col.startswith('prob_')]
    
    if not prob_cols:
        print("❌ Không tìm thấy cột probability nào!")
        return 0
    
    print(f"🗺️ Tạo {len(prob_cols)} cặp file TIFF (probability + risk)...")
    print(f"📁 Thư mục: {os.path.abspath(output_dir)}")
    
    success_count = 0
    
    for col in prob_cols:
        # Tên mô hình từ tên cột
        model_name = col.replace('prob_', '')
        
        # Đường dẫn file TIFF
        prob_tiff_file = os.path.join(output_dir, f'flood_prob_{model_name}.tif')
        risk_tiff_file = os.path.join(output_dir, f'flood_risk_{model_name}.tif')
        
        print(f"\n🔄 Tạo {model_name}...")
        
        # Kiểm tra dữ liệu
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            print(f"   ⚠️ Không có dữ liệu hợp lệ cho {model_name}")
            continue
        
        print(f"   📈 Xác suất: {valid_data.min():.4f} - {valid_data.max():.4f}")
        
        # Tạo probability TIFF
        if create_single_tiff(df, 'x', 'y', col, prob_tiff_file):
            # Tạo risk classification TIFF
            if create_single_tiff(df, 'x', 'y', col, risk_tiff_file, is_risk_map=True):
                success_count += 1
    
    total_files = success_count * 2
    print(f"\n🎉 Đã tạo thành công {total_files} file TIFF ({success_count} models)!")
    print(f"📂 Các file TIFF trong: {os.path.abspath(output_dir)}")
    
    return success_count

def validate_output_directory(output_dir=None):
    """Kiểm tra và tạo thư mục output"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Test write permission
        test_file = os.path.join(output_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        print(f"✅ Thư mục output sẵn sàng: {os.path.abspath(output_dir)}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo thư mục output {output_dir}: {e}")
        return False

def print_tiff_summary(output_dir=None):
    """In tóm tắt các file TIFF đã tạo"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        print(f"❌ Thư mục {output_dir} không tồn tại")
        return
    
    tiff_files = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
    
    if not tiff_files:
        print(f"📂 Chưa có file TIFF nào trong {output_dir}")
        return
    
    print(f"\n📋 DANH SÁCH FILE TIFF ĐÃ TẠO:")
    print("=" * 50)
    
    for i, tiff_file in enumerate(sorted(tiff_files), 1):
        file_path = os.path.join(output_dir, tiff_file)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"{i:2d}. {tiff_file} ({file_size:.1f} MB)")
    
    print("=" * 50)
    print(f"📊 Tổng: {len(tiff_files)} file TIFF")
    print(f"🗺️ Định dạng: GeoTIFF, {CRS}")
    print(f"📐 Độ phân giải: {PIXEL_SIZE}° ≈ {PIXEL_SIZE*111000:.0f}m")
    print(f"💾 Giá trị: Xác suất lũ lụt 0.0-1.0 (NoData={NODATA_VALUE})")