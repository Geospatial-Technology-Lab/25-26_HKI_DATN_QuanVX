import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os
from pathlib import Path
import numpy as np

def cut_tiff(input_tiff, shapefile, output_file):
    """
    Cắt một file TIFF theo shapefile
    """
    try:
        # Đọc shapefile
        shapes = gpd.read_file(shapefile)

        # Đọc file TIFF
        with rasterio.open(input_tiff) as src:
            # Lưu lại dtype gốc để giữ nguyên kiểu dữ liệu
            original_dtype = src.dtypes[0]
            
            # Xác định giá trị NoData phù hợp với kiểu dữ liệu
            if src.nodata is not None:
                nodata_value = src.nodata
            else:
                # Chọn giá trị NoData phù hợp với kiểu dữ liệu
                if 'uint8' in original_dtype:
                    nodata_value = 0  # Dùng 0 thay vì 255 cho risk levels
                elif 'uint16' in original_dtype:
                    nodata_value = 0
                elif 'int' in original_dtype:
                    nodata_value = -9999
                else:
                    nodata_value = -9999.0
            
            # Cắt ảnh theo khu vực shapefile với NoData cho vùng bên ngoài
            out_image, out_transform = mask(
                src, 
                shapes.geometry, 
                crop=True,
                nodata=nodata_value,
                filled=True,
                all_touched=False
            )
            out_meta = src.meta.copy()

        # Cập nhật metadata - GIỮ NGUYÊN dtype gốc
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_value,
            "dtype": original_dtype,  # QUAN TRỌNG: Giữ nguyên dtype gốc
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        })

        # Lưu file TIFF đã cắt
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

        # Kiểm tra giá trị unique sau khi cắt
        unique_values = np.unique(out_image[out_image != nodata_value])
        print(f"✓ Cắt thành công: {os.path.basename(input_tiff)} -> {os.path.basename(output_file)}")
        print(f"  Unique values: {unique_values[:20]}")  # In tối đa 20 giá trị để kiểm tra
        return True
    except Exception as e:
        print(f"✗ Lỗi khi cắt {os.path.basename(input_tiff)}: {e}")
        return False

if __name__ == "__main__":
    # Danh sách các file TIFF đầu vào
    input_tiffs = [
        r"D:\prj\feature\NDBI_2024.tif",
        r"D:\prj\feature\NDWI_2024.tif"
    ]

    # Đường dẫn shapefile để cắt
    shapefile = r"C:\Users\Admin\Desktop\GL\gl.shp"

    print("BẮT ĐẦU CẮT ẢNH TIFF")
    print("="*60)
    print(f"Shapefile: {shapefile}")
    print("="*60)

    # Kiểm tra xem shapefile có tồn tại không
    if not os.path.exists(shapefile):
        print(f"LỖI: Shapefile không tồn tại: {shapefile}")
    else:
        # Xử lý từng file TIFF
        for input_tiff in input_tiffs:
            print(f"\nĐang xử lý: {input_tiff}")
            # Gọi hàm cắt ảnh và ghi đè lên ảnh cũ
            if cut_tiff(input_tiff, shapefile, input_tiff):
                print(f"✓ Hoàn thành cắt ảnh và ghi đè: {os.path.basename(input_tiff)}")
            else:
                print(f"✗ Lỗi khi cắt ảnh: {os.path.basename(input_tiff)}")
        
        print("\n" + "="*60)
        print("HOÀN THÀNH XỬ LÝ TẤT CẢ CÁC FILE")