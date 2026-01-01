"""
Script tính diện tích cho từng ngưỡng của các ảnh TIFF đã phân ngưỡng
Kết quả lưu vào file CSV với:
- Mỗi hàng là một ảnh
- Các cột: Tên ảnh, Ngưỡng 1, Ngưỡng 2, Ngưỡng 3, Ngưỡng 4, Ngưỡng 5, Tổng diện tích
"""

import os
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from datetime import datetime


def tinh_dien_tich_pixel(duong_dan_tiff):
    """
    Tính diện tích pixel cho mỗi ngưỡng trong ảnh TIFF
    
    Args:
        duong_dan_tiff: Đường dẫn đến file TIFF
    
    Returns:
        dict: Dictionary chứa diện tích (m²) cho mỗi ngưỡng
    """
    try:
        with rasterio.open(duong_dan_tiff) as src:
            # Đọc dữ liệu
            data = src.read(1)
            
            # Lấy thông tin transform để tính diện tích pixel
            transform = src.transform
            
            # Tính diện tích 1 pixel (m²)
            # transform[0] là kích thước pixel theo chiều X (longitude)
            # transform[4] là kích thước pixel theo chiều Y (latitude) - thường là âm
            pixel_width = abs(transform[0])  # độ rộng pixel
            pixel_height = abs(transform[4])  # độ cao pixel
            dien_tich_pixel = pixel_width * pixel_height  # diện tích 1 pixel (đơn vị phụ thuộc vào CRS)
            
            # Đếm số pixel cho mỗi ngưỡng
            ket_qua = {
                'Ngưỡng 1 (km²)': 0,
                'Ngưỡng 2 (km²)': 0,
                'Ngưỡng 3 (km²)': 0,
                'Ngưỡng 4 (km²)': 0,
                'Ngưỡng 5 (km²)': 0,
                'Tổng diện tích (km²)': 0
            }
            
            # Đếm pixel cho từng ngưỡng (giá trị 1-5)
            for nguong in range(1, 6):
                so_pixel = np.sum(data == nguong)
                # Chuyển đổi sang km² (giả sử đơn vị là độ decimal degrees)
                # Nếu CRS là UTM hoặc đơn vị mét, cần điều chỉnh công thức
                if src.crs and 'utm' in str(src.crs).lower():
                    # Nếu là UTM, đơn vị là mét
                    dien_tich_km2 = (so_pixel * dien_tich_pixel) / 1_000_000
                else:
                    # Nếu là WGS84 (độ), cần chuyển đổi phức tạp hơn
                    # Ước tính: 1 độ ≈ 111 km ở xích đạo
                    # Công thức đơn giản hóa
                    dien_tich_km2 = (so_pixel * dien_tich_pixel) * (111 * 111)
                
                ket_qua[f'Ngưỡng {nguong} (km²)'] = round(dien_tich_km2, 4)
            
            # Tính tổng diện tích
            ket_qua['Tổng diện tích (km²)'] = round(
                sum(ket_qua[f'Ngưỡng {i} (km²)'] for i in range(1, 6)), 
                4
            )
            
            return ket_qua
            
    except Exception as e:
        print(f"Lỗi khi xử lý {duong_dan_tiff}: {e}")
        return None


def xu_ly_thu_muc(thu_muc_goc, file_csv_output, subfolders=['rf', 'svr', 'xgb'], exclude_folders=['thresholded']):
    """
    Xử lý tất cả các file TIFF trong các thư mục con được chỉ định và tính diện tích
    
    Args:
        thu_muc_goc: Đường dẫn thư mục gốc chứa các thư mục con
        file_csv_output: Đường dẫn file CSV để lưu kết quả
        subfolders: Danh sách các thư mục con cần xử lý
        exclude_folders: Danh sách các thư mục cần bỏ qua
    """
    thu_muc_goc = Path(thu_muc_goc)
    
    if not thu_muc_goc.exists():
        print(f"Thư mục không tồn tại: {thu_muc_goc}")
        return
    
    # Tìm tất cả file TIFF trong các thư mục con được chỉ định
    cac_file_tiff = []
    
    for subfolder in subfolders:
        subfolder_path = thu_muc_goc / subfolder
        
        if not subfolder_path.exists():
            print(f"⚠ Thư mục không tồn tại: {subfolder}")
            continue
        
        print(f"\nQuét thư mục: {subfolder}")
        
        # Tìm file TIFF trong thư mục con
        for file in subfolder_path.glob("*.tif"):
            if not file.name.endswith('.aux.xml'):
                if not any(excluded in str(file) for excluded in exclude_folders):
                    cac_file_tiff.append(file)
        
        for file in subfolder_path.glob("*.tiff"):
            if not file.name.endswith('.aux.xml'):
                if not any(excluded in str(file) for excluded in exclude_folders):
                    cac_file_tiff.append(file)
    
    print(f"\n{'='*80}")
    print(f"Tìm thấy {len(cac_file_tiff)} file TIFF")
    
    # Danh sách để lưu kết quả
    danh_sach_ket_qua = []
    
    # Xử lý từng file
    for file_tiff in sorted(cac_file_tiff):
        print(f"\nĐang xử lý: {file_tiff.name}")
        
        # Tính diện tích
        ket_qua = tinh_dien_tich_pixel(str(file_tiff))
        
        if ket_qua:
            # Lấy tên file và đường dẫn tương đối
            duong_dan_tuong_doi = file_tiff.relative_to(thu_muc_goc)
            ten_day_du = str(duong_dan_tuong_doi).replace('\\', '/')
            
            # Thêm tên file vào kết quả
            ket_qua_row = {
                'Tên ảnh': ten_day_du,
                **ket_qua
            }
            
            danh_sach_ket_qua.append(ket_qua_row)
            
            # In kết quả
            print(f"  Ngưỡng 1: {ket_qua['Ngưỡng 1 (km²)']} km²")
            print(f"  Ngưỡng 2: {ket_qua['Ngưỡng 2 (km²)']} km²")
            print(f"  Ngưỡng 3: {ket_qua['Ngưỡng 3 (km²)']} km²")
            print(f"  Ngưỡng 4: {ket_qua['Ngưỡng 4 (km²)']} km²")
            print(f"  Ngưỡng 5: {ket_qua['Ngưỡng 5 (km²)']} km²")
            print(f"  Tổng: {ket_qua['Tổng diện tích (km²)']} km²")
    
    # Tạo DataFrame và lưu CSV
    if danh_sach_ket_qua:
        df = pd.DataFrame(danh_sach_ket_qua)
        
        # Sắp xếp cột
        cot_sap_xep = [
            'Tên ảnh',
            'Ngưỡng 1 (km²)',
            'Ngưỡng 2 (km²)',
            'Ngưỡng 3 (km²)',
            'Ngưỡng 4 (km²)',
            'Ngưỡng 5 (km²)',
            'Tổng diện tích (km²)'
        ]
        df = df[cot_sap_xep]
        
        # Lưu file CSV
        df.to_csv(file_csv_output, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*80)
        print(f"Đã lưu kết quả vào: {file_csv_output}")
        print("="*80)
        
        # Tính tổng cho tất cả các ảnh
        print("\nTÓM TẮT TỔNG THỂ:")
        print("-" * 80)
        for i in range(1, 6):
            tong = df[f'Ngưỡng {i} (km²)'].sum()
            print(f"Tổng diện tích Ngưỡng {i}: {tong:.4f} km²")
        print(f"Tổng diện tích tất cả: {df['Tổng diện tích (km²)'].sum():.4f} km²")
        print("-" * 80)
        
        # Hiển thị bảng
        print("\nBẢNG KẾT QUẢ:")
        print(df.to_string(index=False))
        
    else:
        print("\nKhông có dữ liệu để lưu!")


if __name__ == "__main__":
    # Đường dẫn thư mục gốc chứa các thư mục con (rf, svr, xgb)
    thu_muc_du_lieu = r"D:\prj\results\map\threshold"
    
    # Các thư mục con cần xử lý
    subfolders = ['rf', 'svr', 'xgb']
    
    # Các thư mục cần bỏ qua
    exclude_folders = ['thresholded']
    
    # Tạo tên file CSV với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_csv_output = f"D:\\prj\\results\\dien_tich_nguong_{timestamp}.csv"
    
    print("="*80)
    print("CHƯƠNG TRÌNH TÍNH DIỆN TÍCH THEO NGƯỠNG")
    print("="*80)
    print(f"Thư mục gốc: {thu_muc_du_lieu}")
    print(f"Thư mục con xử lý: {', '.join(subfolders)}")
    print(f"Thư mục bỏ qua: {', '.join(exclude_folders)}")
    print(f"File kết quả: {file_csv_output}")
    print("="*80)
    print()
    
    # Xử lý thư mục
    xu_ly_thu_muc(thu_muc_du_lieu, file_csv_output, subfolders, exclude_folders)
    
    print("\nHoàn thành!")
