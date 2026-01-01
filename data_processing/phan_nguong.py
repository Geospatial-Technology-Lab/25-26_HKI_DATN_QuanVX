"""
Script phân ngưỡng các ảnh TIFF có giá trị từ 0-1 thành 5 ngưỡng
Ngưỡng 1: 0 - 0.125 -> giá trị 1
Ngưỡng 2: 0.126 - 0.282 -> giá trị 2
Ngưỡng 3: 0.283 - 0.475 -> giá trị 3
Ngưỡng 4: 0.476 - 0.741 -> giá trị 4
Ngưỡng 5: 0.742 - 1.0 -> giá trị 5
"""

import numpy as np
import rasterio
from pathlib import Path


def xu_ly_tiff(duong_dan_dau_vao, duong_dan_dau_ra):
    """
    Đọc file TIFF, phân ngưỡng và lưu kết quả
    
    Args:
        duong_dan_dau_vao: Đường dẫn file TIFF đầu vào
        duong_dan_dau_ra: Đường dẫn file TIFF đầu ra
    """
    print(f"Đang xử lý: {duong_dan_dau_vao}")
    
    # Đọc file TIFF
    try:
        with rasterio.open(duong_dan_dau_vao) as src:
            # Đọc dữ liệu
            data = src.read(1)  # Đọc band đầu tiên
            
            # Lưu metadata
            profile = src.profile.copy()
            
            # Phân ngưỡng
            print("Đang phân ngưỡng...")
            data_phan_nguong = np.zeros_like(data, dtype=np.uint8)
            
            # Xử lý các giá trị hợp lệ (không phải NoData)
            no_data_value = src.nodata
            
            if no_data_value is not None:
                mask = data != no_data_value
            else:
                # Nếu không có NoData, coi giá trị âm hoặc >1 là không hợp lệ
                mask = (data >= 0) & (data <= 1) & (~np.isnan(data))
            
            # Phân ngưỡng cho các giá trị hợp lệ
            data_phan_nguong[mask & (data <= 0.125)] = 1
            data_phan_nguong[mask & (data > 0.125) & (data <= 0.282)] = 2
            data_phan_nguong[mask & (data > 0.282) & (data <= 0.475)] = 3
            data_phan_nguong[mask & (data > 0.475) & (data <= 0.741)] = 4
            data_phan_nguong[mask & (data > 0.741)] = 5
            
            # Giữ nguyên NoData
            data_phan_nguong[~mask] = 0
            
            # Cập nhật profile cho output
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=0
            )
            
            # Ghi file output
            with rasterio.open(duong_dan_dau_ra, 'w', **profile) as dst:
                dst.write(data_phan_nguong, 1)
            
            print(f"Đã lưu: {duong_dan_dau_ra}")
            
    except Exception as e:
        print(f"Không thể đọc file: {duong_dan_dau_vao}")
        print(f"Lỗi: {e}")
        return


def xu_ly_thu_muc(thu_muc_goc, thu_muc_dau_ra, subfolders=['rf', 'svr', 'xgb'], exclude_folders=['thresholded']):
    """
    Xử lý tất cả các file TIFF trong các thư mục con được chỉ định và lưu vào thư mục đầu ra
    
    Args:
        thu_muc_goc: Đường dẫn thư mục gốc chứa các thư mục con
        thu_muc_dau_ra: Đường dẫn thư mục đầu ra
        subfolders: Danh sách các thư mục con cần xử lý
        exclude_folders: Danh sách các thư mục cần bỏ qua
    """
    thu_muc_goc = Path(thu_muc_goc)
    thu_muc_dau_ra = Path(thu_muc_dau_ra)
    
    if not thu_muc_goc.exists():
        print(f"Thư mục không tồn tại: {thu_muc_goc}")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    thu_muc_dau_ra.mkdir(parents=True, exist_ok=True)
    print(f"Thư mục đầu ra: {thu_muc_dau_ra}")
    
    # Đếm số file đã xử lý
    success_count = 0
    fail_count = 0
    total_count = 0
    
    # Duyệt qua các thư mục con được chỉ định
    for subfolder in subfolders:
        subfolder_path = thu_muc_goc / subfolder
        
        if not subfolder_path.exists():
            print(f"⚠ Thư mục không tồn tại: {subfolder}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Đang xử lý thư mục: {subfolder}")
        print(f"{'='*60}")
        
        # Tìm tất cả file TIFF trong thư mục con
        cac_file_tiff = []
        for file in subfolder_path.glob("*.tif"):
            # Bỏ qua file .aux.xml và các file trong thư mục exclude
            if not file.name.endswith('.aux.xml'):
                if not any(excluded in str(file) for excluded in exclude_folders):
                    cac_file_tiff.append(file)
        
        cac_file_tiff.extend([f for f in subfolder_path.glob("*.tiff") 
                             if not f.name.endswith('.aux.xml') 
                             and not any(excluded in str(f) for excluded in exclude_folders)])
        
        print(f"Tìm thấy {len(cac_file_tiff)} file TIFF trong {subfolder}")
        
        # Tạo thư mục con trong thư mục đầu ra
        subfolder_dau_ra = thu_muc_dau_ra / subfolder
        subfolder_dau_ra.mkdir(parents=True, exist_ok=True)
        
        # Xử lý từng file
        for file_tiff in cac_file_tiff:
            total_count += 1
            
            print(f"\n[{total_count}] Đang xử lý: {subfolder}/{file_tiff.name}")
            
            # Tạo đường dẫn file đầu ra
            file_dau_ra = subfolder_dau_ra / file_tiff.name
            
            try:
                # Phân ngưỡng và lưu vào file đầu ra
                xu_ly_tiff(str(file_tiff), str(file_dau_ra))
                print(f"  → Đã lưu vào: {subfolder}/threshold/{file_tiff.name}")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Lỗi khi xử lý {file_tiff.name}: {e}")
                fail_count += 1
    
    # Tổng kết
    print(f"\n{'='*60}")
    print("TỔNG KẾT:")
    print(f"  Tổng số file: {total_count}")
    print(f"  Thành công: {success_count}")
    print(f"  Thất bại: {fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # File TIFF đầu vào
    file_dau_vao = r"D:\prj\results\map\xgb\flood_susceptibility_po_XGB.tif"
    
    # File đầu ra
    file_dau_ra = r"D:\prj\results\map\threshold\xgb\flood_susceptibility_po_XGB.tif"
    
    print("="*60)
    print("CHƯƠNG TRÌNH PHÂN NGƯỠNG ẢNH TIFF")
    print("="*60)
    print(f"File đầu vào: {file_dau_vao}")
    print(f"File đầu ra: {file_dau_ra}")
    print("Phân ngưỡng:")
    print("  - Ngưỡng 1: 0.000 - 0.125  -> Giá trị 1")
    print("  - Ngưỡng 2: 0.126 - 0.282  -> Giá trị 2")
    print("  - Ngưỡng 3: 0.283 - 0.475  -> Giá trị 3")
    print("  - Ngưỡng 4: 0.476 - 0.741  -> Giá trị 4")
    print("  - Ngưỡng 5: 0.742 - 1.000  -> Giá trị 5")
    print("="*60)
    print()
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    from pathlib import Path
    Path(file_dau_ra).parent.mkdir(parents=True, exist_ok=True)
    
    # Xử lý file
    xu_ly_tiff(file_dau_vao, file_dau_ra)
