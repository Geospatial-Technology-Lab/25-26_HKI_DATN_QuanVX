import rasterio
import numpy as np
import jenkspy

# Đường dẫn file
input_file = r"D:\prj\feature\gialai_curvature.tif"
output_file = r"D:\prj\feature\gialai_curvature_natural_breaks.tif"

print(f"{'='*60}")
print(f"TÌM NATURAL BREAKS VÀ LỌC DỮ LIỆU")
print(f"{'='*60}")

with rasterio.open(input_file) as src:
    data = src.read(1)
    profile = src.profile.copy()
    
    # Lấy dữ liệu hợp lệ
    valid_mask = ~np.isnan(data)
    if src.nodata is not None:
        valid_mask = valid_mask & (data != src.nodata)
    
    valid_data = data[valid_mask]
    
    print(f"\nDữ liệu gốc:")
    print(f"  Tổng số pixels: {data.size:,}")
    print(f"  Pixels hợp lệ:  {len(valid_data):,} ({len(valid_data)/data.size*100:.2f}%)")
    print(f"  Min: {valid_data.min():.2f}, Max: {valid_data.max():.2f}")
    
    # Lấy mẫu ngẫu nhiên để tính Natural Breaks (vì dữ liệu quá lớn)
    sample_size = min(100000, len(valid_data))
    sample_indices = np.random.choice(len(valid_data), sample_size, replace=False)
    sample_data = valid_data[sample_indices]
    
    print(f"\n{'='*60}")
    print(f"TÍNH NATURAL BREAKS (JENKS)")
    print(f"{'='*60}")
    print(f"Sử dụng {sample_size:,} mẫu ngẫu nhiên...")
    
    # Tính Natural Breaks với số lớp khác nhau
    for n_classes in [3, 5, 7]:
        print(f"\n{n_classes} lớp:")
        breaks = jenkspy.jenks_breaks(sample_data, n_classes=n_classes)
        
        print(f"  Điểm gãy: ", end="")
        for i, b in enumerate(breaks):
            if i < len(breaks) - 1:
                print(f"{b:.2f}", end=" → ")
            else:
                print(f"{b:.2f}")
        
        # Hiển thị phân bố theo các lớp
        for i in range(len(breaks) - 1):
            mask = (valid_data >= breaks[i]) & (valid_data < breaks[i+1])
            count = np.sum(mask)
            percent = count / len(valid_data) * 100
            print(f"  Lớp {i+1} [{breaks[i]:.2f}, {breaks[i+1]:.2f}): {count:,} ({percent:.2f}%)")
    
    # Sử dụng 5 lớp làm mặc định
    n_classes = 5
    breaks = jenkspy.jenks_breaks(sample_data, n_classes=n_classes)
    
    print(f"\n{'='*60}")
    print(f"LỌC DỮ LIỆU (Sử dụng {n_classes} lớp)")
    print(f"{'='*60}")
    
    # Loại bỏ lớp đầu và lớp cuối (giữ lại 3 lớp giữa)
    lower_bound = breaks[1]  # Bỏ lớp 1
    upper_bound = breaks[-2]  # Bỏ lớp cuối
    
    print(f"\nKhoảng giá trị giữ lại: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Đếm số lượng giá trị sẽ giữ lại
    keep_mask = (valid_data >= lower_bound) & (valid_data < upper_bound)
    keep_count = np.sum(keep_mask)
    remove_count = len(valid_data) - keep_count
    
    print(f"\nDự kiến:")
    print(f"  Giữ lại:  {keep_count:,} pixels ({keep_count/len(valid_data)*100:.2f}%)")
    print(f"  Loại bỏ:  {remove_count:,} pixels ({remove_count/len(valid_data)*100:.2f}%)")
    
    # Tạo dữ liệu mới
    filtered_data = np.full_like(data, np.nan, dtype=np.float32)
    
    # Chỉ copy các giá trị nằm trong khoảng
    keep_full_mask = valid_mask & (data >= lower_bound) & (data < upper_bound)
    filtered_data[keep_full_mask] = data[keep_full_mask]
    
    # Thống kê sau lọc
    final_valid_mask = ~np.isnan(filtered_data)
    final_valid_data = filtered_data[final_valid_mask]
    
    print(f"\nKết quả thực tế:")
    print(f"  Còn lại:    {len(final_valid_data):,} pixels")
    print(f"\nThống kê dữ liệu còn lại:")
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
