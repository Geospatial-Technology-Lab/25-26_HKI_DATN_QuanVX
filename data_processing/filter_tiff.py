import rasterio
import numpy as np
from rasterio.transform import from_bounds

# Đường dẫn file đầu vào và đầu ra
input_file = r"D:\prj\feature\gialai_curvature.tif"
output_file = r"D:\prj\feature\gialai_curvature_filtered.tif"

print(f"{'='*60}")
print(f"LỌC DỮ LIỆU CURVATURE")
print(f"{'='*60}")
print(f"Input:  {input_file}")
print(f"Output: {output_file}")

with rasterio.open(input_file) as src:
    data = src.read(1)
    profile = src.profile.copy()
    
    print(f"\n{'='*60}")
    print(f"DỮ LIỆU GỐC")
    print(f"{'='*60}")
    print(f"Kích thước: {data.shape[0]} x {data.shape[1]} = {data.size:,} pixels")
    
    # Đếm giá trị null/nodata ban đầu
    initial_null_mask = np.isnan(data)
    if src.nodata is not None:
        initial_null_mask = initial_null_mask | (data == src.nodata)
    
    initial_null_count = np.sum(initial_null_mask)
    initial_valid_count = data.size - initial_null_count
    
    print(f"Giá trị hợp lệ: {initial_valid_count:,} ({initial_valid_count/data.size*100:.2f}%)")
    print(f"Giá trị null:   {initial_null_count:,} ({initial_null_count/data.size*100:.2f}%)")
    
    # Tạo mask cho dữ liệu hợp lệ (không null và trong khoảng [-1, 1])
    valid_mask = ~initial_null_mask
    valid_data = data[valid_mask]
    
    # Đếm giá trị ngoài khoảng [-1, 1]
    outside_range = (valid_data < -1) | (valid_data > 1)
    outside_count = np.sum(outside_range)
    
    print(f"\nGiá trị ngoài khoảng [-1, 1]: {outside_count:,} ({outside_count/initial_valid_count*100:.2f}% của dữ liệu hợp lệ)")
    
    # Tạo dữ liệu mới: giữ nguyên giá trị trong khoảng [-1, 1], còn lại đặt thành nodata
    filtered_data = data.copy()
    
    # Đặt các giá trị ngoài khoảng [-1, 1] thành nodata
    out_of_range_mask = (data < -30) | (data > 30)
    filtered_data[out_of_range_mask] = np.nan
    
    # Đếm sau khi lọc
    final_null_mask = np.isnan(filtered_data)
    final_null_count = np.sum(final_null_mask)
    final_valid_count = data.size - final_null_count
    
    print(f"\n{'='*60}")
    print(f"DỮ LIỆU SAU KHI LỌC")
    print(f"{'='*60}")
    print(f"Giá trị hợp lệ (trong [-1, 1]): {final_valid_count:,} ({final_valid_count/data.size*100:.2f}%)")
    print(f"Giá trị null/nodata:            {final_null_count:,} ({final_null_count/data.size*100:.2f}%)")
    print(f"Đã loại bỏ:                     {outside_count:,} giá trị ngoài khoảng")
    
    # Thống kê dữ liệu sau lọc
    remaining_valid = filtered_data[~np.isnan(filtered_data)]
    if len(remaining_valid) > 0:
        print(f"\nThống kê dữ liệu còn lại:")
        print(f"  Min:    {remaining_valid.min():.6f}")
        print(f"  Max:    {remaining_valid.max():.6f}")
        print(f"  Mean:   {remaining_valid.mean():.6f}")
        print(f"  Median: {np.median(remaining_valid):.6f}")
        print(f"  Std:    {np.std(remaining_valid):.6f}")
    
    # Cập nhật profile cho file đầu ra
    profile.update(
        dtype=rasterio.float32,
        nodata=np.nan
    )
    
    # Ghi file đầu ra
    print(f"\n{'='*60}")
    print(f"Đang ghi file...")
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(filtered_data, 1)
    
    print(f"✓ Đã lưu file: {output_file}")
    print(f"{'='*60}")
