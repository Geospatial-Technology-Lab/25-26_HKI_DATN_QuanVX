import rasterio
import numpy as np

file_path = r"D:\prj\feature\gialai_curvature.tif"

with rasterio.open(file_path) as ds:
    data = ds.read(1)
    
    print(f"{'='*60}")
    print(f"PHÂN TÍCH DỮ LIỆU CURVATURE - ĐỘ CONG ĐỊA HÌNH")
    print(f"{'='*60}")
    
    # Thông tin cơ bản
    print(f"\nThông tin file:")
    print(f"  Data type: {data.dtype}")
    print(f"  Data shape: {data.shape[0]} x {data.shape[1]} = {data.size:,} pixels")
    print(f"  CRS: {ds.crs}")
    
    # Xử lý NoData và NaN
    valid_mask = ~np.isnan(data)
    if ds.nodata is not None:
        valid_mask = valid_mask & (data != ds.nodata)
    
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        print("\n⚠ CẢNH BÁO: Không có dữ liệu hợp lệ trong file!")
        exit()
    
    # Thống kê dữ liệu hợp lệ
    print(f"\nDữ liệu hợp lệ:")
    print(f"  Số pixel hợp lệ: {len(valid_data):,} ({len(valid_data)/data.size*100:.2f}%)")
    print(f"  Số pixel không hợp lệ: {data.size - len(valid_data):,} ({(data.size - len(valid_data))/data.size*100:.2f}%)")
    
    # Dải giá trị
    print(f"\n{'='*60}")
    print(f"DẢI GIÁ TRỊ CURVATURE")
    print(f"{'='*60}")
    print(f"  Min:    {valid_data.min():.6f}")
    print(f"  Max:    {valid_data.max():.6f}")
    print(f"  Mean:   {valid_data.mean():.6f}")
    print(f"  Median: {np.median(valid_data):.6f}")
    print(f"  Std:    {np.std(valid_data):.6f}")
    
    # Phân tích giá trị nằm ngoài các khoảng
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH GIÁ TRỊ NGOÀI KHOẢNG")
    print(f"{'='*60}")
    
    # Khoảng [-1, 1]
    outside_1 = (valid_data < -1) | (valid_data > 1)
    count_outside_1 = np.sum(outside_1)
    print(f"\nNgoài khoảng [-1, 1]:")
    print(f"  Số lượng: {count_outside_1:,} pixels")
    print(f"  Tỷ lệ:    {count_outside_1/len(valid_data)*100:.2f}%")
    
    # Khoảng [-30, 30]
    outside_30 = (valid_data < -30) | (valid_data > 30)
    count_outside_30 = np.sum(outside_30)
    print(f"\nNgoài khoảng [-30, 30]:")
    print(f"  Số lượng: {count_outside_30:,} pixels")
    print(f"  Tỷ lệ:    {count_outside_30/len(valid_data)*100:.2f}%")
    
    # Phân bố chi tiết theo khoảng giá trị
    print(f"\n{'='*60}")
    print(f"PHÂN BỐ CHI TIẾT THEO KHOẢNG GIÁ TRỊ")
    print(f"{'='*60}")
    
    ranges = [
        ("< -100", valid_data < -100),
        ("[-100, -50)", (valid_data >= -100) & (valid_data < -50)),
        ("[-50, -30)", (valid_data >= -50) & (valid_data < -30)),
        ("[-30, -10)", (valid_data >= -30) & (valid_data < -10)),
        ("[-10, -5)", (valid_data >= -10) & (valid_data < -5)),
        ("[-5, -1)", (valid_data >= -5) & (valid_data < -1)),
        ("[-1, -0.5)", (valid_data >= -1) & (valid_data < -0.5)),
        ("[-0.5, 0)", (valid_data >= -0.5) & (valid_data < 0)),
        ("[0, 0.5)", (valid_data >= 0) & (valid_data < 0.5)),
        ("[0.5, 1)", (valid_data >= 0.5) & (valid_data < 1)),
        ("[1, 5)", (valid_data >= 1) & (valid_data < 5)),
        ("[5, 10)", (valid_data >= 5) & (valid_data < 10)),
        ("[10, 30)", (valid_data >= 10) & (valid_data < 30)),
        ("[30, 50)", (valid_data >= 30) & (valid_data < 50)),
        ("[50, 100)", (valid_data >= 50) & (valid_data < 100)),
        (">= 100", valid_data >= 100),
    ]
    
    print(f"\n{'Khoảng giá trị':<20} {'Số lượng':>15} {'Tỷ lệ %':>10}")
    print(f"{'-'*20} {'-'*15} {'-'*10}")
    
    for range_name, mask in ranges:
        count = np.sum(mask)
        percent = count / len(valid_data) * 100
        if count > 0:
            print(f"{range_name:<20} {count:>15,} {percent:>9.2f}%")
    
    # Percentiles
    print(f"\n{'='*60}")
    print(f"PHÂN VỊ (PERCENTILES)")
    print(f"{'='*60}")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n{'Percentile':<15} {'Giá trị':>15}")
    print(f"{'-'*15} {'-'*15}")
    for p in percentiles:
        value = np.percentile(valid_data, p)
        print(f"{p}%{'':<12} {value:>15.6f}")
    
    print(f"\n{'='*60}")