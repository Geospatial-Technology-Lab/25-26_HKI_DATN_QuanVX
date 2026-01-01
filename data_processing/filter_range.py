import rasterio
import numpy as np

# Đường dẫn file
input_file = r"D:\prj\feature\gialai_curvature.tif"
output_file = r"D:\prj\feature\gialai_curvature_filtered_range.tif"

# Khoảng giá trị giữ lại
lower_bound = -24
upper_bound = 18

print(f"{'='*60}")
print(f"LỌC DỮ LIỆU THEO KHOẢNG GIÁ TRỊ")
print(f"{'='*60}")
print(f"Input:  {input_file}")
print(f"Output: {output_file}")
print(f"Khoảng giữ lại: [{lower_bound}, {upper_bound}]")

with rasterio.open(input_file) as src:
    data = src.read(1)
    profile = src.profile.copy()
    
    # Lấy dữ liệu hợp lệ
    valid_mask = ~np.isnan(data)
    if src.nodata is not None:
        valid_mask = valid_mask & (data != src.nodata)
    
    valid_data = data[valid_mask]
    
    print(f"\n{'='*60}")
    print(f"DỮ LIỆU GỐC")
    print(f"{'='*60}")
    print(f"Tổng số pixels:  {data.size:,}")
    print(f"Pixels hợp lệ:   {len(valid_data):,} ({len(valid_data)/data.size*100:.2f}%)")
    print(f"Pixels null:     {data.size - len(valid_data):,} ({(data.size - len(valid_data))/data.size*100:.2f}%)")
    print(f"\nDải giá trị:")
    print(f"  Min:    {valid_data.min():.2f}")
    print(f"  Max:    {valid_data.max():.2f}")
    print(f"  Mean:   {valid_data.mean():.2f}")
    print(f"  Median: {np.median(valid_data):.2f}")
    print(f"  Std:    {np.std(valid_data):.2f}")
    
    # Phân tích giá trị sẽ bị loại bỏ
    below_lower = valid_data < lower_bound
    above_upper = valid_data > upper_bound
    in_range = (valid_data >= lower_bound) & (valid_data <= upper_bound)
    
    count_below = np.sum(below_lower)
    count_above = np.sum(above_upper)
    count_keep = np.sum(in_range)
    
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH")
    print(f"{'='*60}")
    print(f"Giá trị < {lower_bound}:  {count_below:,} ({count_below/len(valid_data)*100:.2f}%)")
    print(f"Giá trị > {upper_bound}:   {count_above:,} ({count_above/len(valid_data)*100:.2f}%)")
    print(f"Giá trị trong [{lower_bound}, {upper_bound}]: {count_keep:,} ({count_keep/len(valid_data)*100:.2f}%)")
    print(f"\nTổng loại bỏ: {count_below + count_above:,} ({(count_below + count_above)/len(valid_data)*100:.2f}%)")
    
    # Tạo dữ liệu mới
    filtered_data = np.full_like(data, np.nan, dtype=np.float32)
    
    # Chỉ giữ lại các giá trị trong khoảng
    keep_mask = valid_mask & (data >= lower_bound) & (data <= upper_bound)
    filtered_data[keep_mask] = data[keep_mask]
    
    # Thống kê sau lọc
    final_valid_mask = ~np.isnan(filtered_data)
    final_valid_data = filtered_data[final_valid_mask]
    
    print(f"\n{'='*60}")
    print(f"DỮ LIỆU SAU KHI LỌC")
    print(f"{'='*60}")
    print(f"Pixels hợp lệ:   {len(final_valid_data):,} ({len(final_valid_data)/data.size*100:.2f}%)")
    print(f"Pixels null:     {data.size - len(final_valid_data):,} ({(data.size - len(final_valid_data))/data.size*100:.2f}%)")
    print(f"\nDải giá trị:")
    print(f"  Min:    {final_valid_data.min():.2f}")
    print(f"  Max:    {final_valid_data.max():.2f}")
    print(f"  Mean:   {final_valid_data.mean():.2f}")
    print(f"  Median: {np.median(final_valid_data):.2f}")
    print(f"  Std:    {np.std(final_valid_data):.2f}")
    
    # Cập nhật profile
    profile.update(
        dtype=rasterio.float32,
        nodata=np.nan
    )
    
    # Ghi file
    print(f"\n{'='*60}")
    print(f"Đang ghi file...")
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(filtered_data, 1)
    
    print(f"✓ Đã lưu: {output_file}")
    print(f"{'='*60}")
