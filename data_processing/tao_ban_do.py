import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import rasterio
import contextily as ctx
import os
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer

# ===== ĐỌC FILE TIFF VÀ CHUYỂN ĐỔI HỆ TỌA ĐỘ =====
def read_tiff(file_path):
    with rasterio.open(file_path) as ds:
        data = ds.read(1)  # Đọc band đầu tiên
        bounds = ds.bounds
        src_crs = ds.crs  # Lấy hệ tọa độ gốc
        nodata_value = ds.nodata  # Lấy giá trị NoData từ metadata
        
        print(f"  - NoData value từ file: {nodata_value}")
        print(f"  - Data min: {data.min()}, max: {data.max()}")
        print(f"  - Data dtype: {data.dtype}")
        
        # extent gốc cho vẽ data
        extent_original = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # Chuyển đổi bounds sang WGS84 (EPSG:4326) cho lưới tọa độ
        transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True) if src_crs and src_crs.to_string() != 'EPSG:4326' else None
        if transformer:
            x_min_wgs, y_min_wgs = transformer.transform(bounds.left, bounds.bottom)
            x_max_wgs, y_max_wgs = transformer.transform(bounds.right, bounds.top)
        else:
            x_min_wgs, y_min_wgs, x_max_wgs, y_max_wgs = bounds.left, bounds.bottom, bounds.right, bounds.top
        extent_wgs84 = [x_min_wgs, x_max_wgs, y_min_wgs, y_max_wgs]
    
    return data, extent_original, extent_wgs84, src_crs, nodata_value

# ===== HÀM TẠO BỘ MÀU CHO PHÂN NGƯỠNG =====
def get_threshold_colormap():
    """Trả về colormap cho dữ liệu phân ngưỡng (1-5)"""
    from matplotlib.colors import ListedColormap
    
    # Màu cho từng mức ngưỡng (0: NoData, 1-5: các mức)
    colors = [
        '#FFFFFF',  # 0 - NoData (trắng/transparent)
        '#55FF00',  # 1 - Rất thấp
        '#AAFF00',  # 2 - Thấp
        '#FFFF00',  # 3 - Trung bình
        '#FFAA00',  # 4 - Cao
        '#FF0000'   # 5 - Rất cao
    ]
    
    label = 'Mức độ nhạy cảm với ngập lụt'
    cmap = ListedColormap(colors)
    cmap.set_bad(color='lightgray', alpha=0.3)
    return cmap, label

# ===== HÀM VẼ LA BÀN CẢI TIẾN =====
def draw_compass_rose(ax, x_pos=0.95, y_pos=0.88, size=0.045):
    """
    Vẽ la bàn với thiết kế chi tiết giống ảnh mẫu
    
    Parameters:
    - ax: matplotlib axes
    - x_pos, y_pos: vị trí la bàn (tọa độ transform)
    - size: kích thước la bàn
    """
    # Vòng tròn ngoài cùng (viền đôi) - giảm độ đậm
    outer_circle1 = Circle((x_pos, y_pos), size, transform=ax.transAxes,
                          color='white', ec='black', linewidth=1.2, zorder=10, alpha=1.0)
    ax.add_patch(outer_circle1)
    
    outer_circle2 = Circle((x_pos, y_pos), size*0.93, transform=ax.transAxes,
                          fill=False, ec='black', linewidth=0.8, zorder=10)
    ax.add_patch(outer_circle2)
    
    # Vòng tròn thứ ba (trong)
    inner_circle = Circle((x_pos, y_pos), size*0.80, transform=ax.transAxes,
                         fill=False, ec='#666', linewidth=0.8, zorder=10, alpha=0.7)
    ax.add_patch(inner_circle)
    
    # Vẽ các vạch chia độ giữa 2 vòng ngoài
    for i in range(72):
        angle = i * 5  # Mỗi 5 độ
        angle_rad = np.radians(angle)
        
        # Xác định loại vạch
        is_cardinal = i % 18 == 0  # N, E, S, W (mỗi 90 độ)
        is_major = i % 6 == 0      # Mỗi 30 độ
        is_minor = i % 2 == 0      # Mỗi 10 độ
        
        if is_cardinal:
            continue  # Bỏ qua vị trí 4 hướng chính
        
        # Xác định độ dài vạch
        if is_major:
            inner_radius = size * 0.84
            outer_radius = size * 0.93
            linewidth = 1.2
        elif is_minor:
            inner_radius = size * 0.89
            outer_radius = size * 0.93
            linewidth = 0.8
        else:
            inner_radius = size * 0.91
            outer_radius = size * 0.93
            linewidth = 0.5
        
        # Tính tọa độ
        x_start = x_pos + inner_radius * np.sin(angle_rad)
        y_start = y_pos + inner_radius * np.cos(angle_rad)
        x_end = x_pos + outer_radius * np.sin(angle_rad)
        y_end = y_pos + outer_radius * np.cos(angle_rad)
        
        ax.plot([x_start, x_end], [y_start, y_end],
               transform=ax.transAxes, color='black', linewidth=linewidth, zorder=11)
    
    # Vẽ 8 đường từ tâm ra (4 chính + 4 phụ)
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        angle_rad = np.radians(angle)
        is_cardinal = angle % 90 == 0
        
        x_start = x_pos + size * 0.13 * np.sin(angle_rad)
        y_start = y_pos + size * 0.13 * np.cos(angle_rad)
        x_end = x_pos + size * 0.80 * np.sin(angle_rad)
        y_end = y_pos + size * 0.80 * np.cos(angle_rad)
        
        linewidth = 1.5 if is_cardinal else 1.0
        
        ax.plot([x_start, x_end], [y_start, y_end],
               transform=ax.transAxes, color='black', 
               linewidth=linewidth, zorder=11, alpha=0.8)
    
    # Vẽ 8 cánh ngôi sao (4 chính + 4 phụ)
    # Cánh Bắc (N) - đen đậm
    north_arrow = Polygon([
        (x_pos, y_pos + size * 0.80),
        (x_pos - size * 0.066, y_pos + size * 0.13),
        (x_pos, y_pos),
        (x_pos + size * 0.066, y_pos + size * 0.13)
    ], transform=ax.transAxes, facecolor='#000', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(north_arrow)
    
    # Cánh Đông Bắc (NE) - xám
    ne_arrow = Polygon([
        (x_pos + size * 0.566, y_pos + size * 0.566),
        (x_pos + size * 0.092, y_pos + size * 0.092),
        (x_pos, y_pos),
        (x_pos + size * 0.092, y_pos + size * 0.046)
    ], transform=ax.transAxes, facecolor='#666', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(ne_arrow)
    
    # Cánh Đông (E) - xám nhạt
    east_arrow = Polygon([
        (x_pos + size * 0.80, y_pos),
        (x_pos + size * 0.13, y_pos + size * 0.066),
        (x_pos, y_pos),
        (x_pos + size * 0.13, y_pos - size * 0.066)
    ], transform=ax.transAxes, facecolor='#999', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(east_arrow)
    
    # Cánh Đông Nam (SE) - xám
    se_arrow = Polygon([
        (x_pos + size * 0.566, y_pos - size * 0.566),
        (x_pos + size * 0.092, y_pos - size * 0.046),
        (x_pos, y_pos),
        (x_pos + size * 0.092, y_pos - size * 0.092)
    ], transform=ax.transAxes, facecolor='#666', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(se_arrow)
    
    # Cánh Nam (S) - xám nhạt
    south_arrow = Polygon([
        (x_pos, y_pos - size * 0.80),
        (x_pos + size * 0.066, y_pos - size * 0.13),
        (x_pos, y_pos),
        (x_pos - size * 0.066, y_pos - size * 0.13)
    ], transform=ax.transAxes, facecolor='#999', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(south_arrow)
    
    # Cánh Tây Nam (SW) - xám
    sw_arrow = Polygon([
        (x_pos - size * 0.566, y_pos - size * 0.566),
        (x_pos - size * 0.092, y_pos - size * 0.092),
        (x_pos, y_pos),
        (x_pos - size * 0.092, y_pos - size * 0.046)
    ], transform=ax.transAxes, facecolor='#666', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(sw_arrow)
    
    # Cánh Tây (W) - xám nhạt
    west_arrow = Polygon([
        (x_pos - size * 0.80, y_pos),
        (x_pos - size * 0.13, y_pos - size * 0.066),
        (x_pos, y_pos),
        (x_pos - size * 0.13, y_pos + size * 0.066)
    ], transform=ax.transAxes, facecolor='#999', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(west_arrow)
    
    # Cánh Tây Bắc (NW) - xám
    nw_arrow = Polygon([
        (x_pos - size * 0.566, y_pos + size * 0.566),
        (x_pos - size * 0.092, y_pos + size * 0.046),
        (x_pos, y_pos),
        (x_pos - size * 0.092, y_pos + size * 0.092)
    ], transform=ax.transAxes, facecolor='#666', edgecolor='black', linewidth=0.5, zorder=12)
    ax.add_patch(nw_arrow)
    
    # Vòng tròn trung tâm lớn (trắng)
    center_circle = Circle((x_pos, y_pos), size*0.11, transform=ax.transAxes,
                          color='white', ec='black', linewidth=1.5, zorder=13)
    ax.add_patch(center_circle)
    
    # Chấm đen nhỏ ở giữa
    center_dot = Circle((x_pos, y_pos), size*0.04, transform=ax.transAxes,
                       color='black', ec='black', linewidth=0.5, zorder=14)
    ax.add_patch(center_dot)
    
    # Chữ hướng (N, E, S, W) - đặt xa hơn bên ngoài la bàn
    ax.text(x_pos, y_pos + size * 1.30, 'N', transform=ax.transAxes,
           ha='center', va='center', fontsize=16, fontweight='bold', zorder=15,
           family='serif', color='black')
    ax.text(x_pos + size * 1.30, y_pos, 'E', transform=ax.transAxes,
           ha='center', va='center', fontsize=16, fontweight='bold', zorder=15,
           family='serif', color='black')
    ax.text(x_pos, y_pos - size * 1.30, 'S', transform=ax.transAxes,
           ha='center', va='center', fontsize=16, fontweight='bold', zorder=15,
           family='serif', color='black')
    ax.text(x_pos - size * 1.30, y_pos, 'W', transform=ax.transAxes,
           ha='center', va='center', fontsize=16, fontweight='bold', zorder=15,
           family='serif', color='black')

# ===== CẤU HÌNH THƯ MỤC =====
input_folder = r"D:\prj\results\map\threshold"
output_folder = r"D:\prj\map\ThreshHold"

# Các thư mục con cần xử lý
subfolders = ['rf', 'svr', 'xgb']

# Các thư mục cần bỏ qua
exclude_folders = ['thresholded']

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# ===== TÌM TẤT CẢ FILE TIFF =====
print(f"Đang tìm kiếm file TIFF trong: {input_folder}")
print(f"Thư mục con xử lý: {', '.join(subfolders)}")
print(f"Thư mục bỏ qua: {', '.join(exclude_folders)}")
print("="*80)

# Tìm tất cả file .tif trong các thư mục con được chỉ định
all_tiff_files = []
for subfolder in subfolders:
    subfolder_path = os.path.join(input_folder, subfolder)
    
    if not os.path.exists(subfolder_path):
        print(f"⚠ Thư mục không tồn tại: {subfolder}")
        continue
    
    print(f"\nQuét thư mục: {subfolder}")
    for file in os.listdir(subfolder_path):
        if (file.endswith('.tif') or file.endswith('.tiff')) and not file.endswith('.aux.xml'):
            # Bỏ qua file trong thư mục exclude
            full_path = os.path.join(subfolder_path, file)
            if not any(excluded in full_path for excluded in exclude_folders):
                all_tiff_files.append(full_path)
                print(f"  - {file}")

print("\n" + "="*80)
print(f"Tìm thấy {len(all_tiff_files)} file TIFF:")
for i, file in enumerate(all_tiff_files, 1):
    print(f"  {i}. {os.path.basename(file)}")
print("="*80)

# ===== XỬ LÝ TỪNG FILE =====
for idx, input_file in enumerate(all_tiff_files, 1):
    # Phân tích tên file để tạo tên output
    filename = os.path.basename(input_file)
    
    # Trích xuất algorithm và model từ tên file
    try:
        # Loại bỏ các tiền tố và hậu tố không cần thiết
        base_name = filename.replace('cliped_', '').replace('_thresholded', '').replace('.tif', '').replace('.tiff', '')
        # Lấy phần flood_susceptibility_xxx_YYY hoặc flood_probability_xxx_YYY
        parts = base_name.replace('flood_susceptibility_', '').replace('flood_probability_', '').split('_')
        if len(parts) >= 2:
            algorithm = parts[0]  # pso, puma, rso
            model = parts[1]      # RF, SVR, XGB
            output_filename = f"{algorithm}_{model}.png"
        else:
            output_filename = filename.replace('.tif', '.png').replace('.tiff', '.png')
    except:
        output_filename = filename.replace('.tif', '.png').replace('.tiff', '.png')
    
    output_file = os.path.join(output_folder, output_filename)

    print(f"\n{'='*80}")
    print(f"[{idx}/{len(all_tiff_files)}] ĐANG XỬ LÝ: {filename}")
    print(f"{'='*80}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    try:
        print("Đọc file TIFF...")
        data, extent_original, extent_wgs84, src_crs, nodata_value = read_tiff(input_file)
        
        print(f"  - Dữ liệu phân ngưỡng (0 = NoData, 1-5 = mức độ nhạy cảm)")
        
        # Mask NoData (giá trị 0) và NaN
        mask_condition = np.isnan(data) | (data == 0)
        if nodata_value is not None:
            mask_condition = mask_condition | (data == nodata_value)
        print(f"  - Masking NaN và NoData values")
        data_masked = np.ma.masked_where(mask_condition, data)

        # Kiểm tra dữ liệu
        valid_data = data_masked.compressed()
        
        if len(valid_data) > 0:
            print(f"Data range (valid): {int(valid_data.min())} to {int(valid_data.max())}")
            
            # Thống kê theo ngưỡng
            level_1 = np.sum(valid_data == 1)
            level_2 = np.sum(valid_data == 2)
            level_3 = np.sum(valid_data == 3)
            level_4 = np.sum(valid_data == 4)
            level_5 = np.sum(valid_data == 5)
            total = len(valid_data)
            
            print(f"  - Phân bố mức độ nhạy cảm:")
            print(f"    + Mức 1 (Rất thấp, 0.000-0.125): {level_1:,} pixels ({level_1/total*100:.2f}%)")
            print(f"    + Mức 2 (Thấp, 0.126-0.282): {level_2:,} pixels ({level_2/total*100:.2f}%)")
            print(f"    + Mức 3 (Trung bình, 0.283-0.475): {level_3:,} pixels ({level_3/total*100:.2f}%)")
            print(f"    + Mức 4 (Cao, 0.476-0.741): {level_4:,} pixels ({level_4/total*100:.2f}%)")
            print(f"    + Mức 5 (Rất cao, 0.742-1.000): {level_5:,} pixels ({level_5/total*100:.2f}%)")
        else:
            print("WARNING: Không có dữ liệu hợp lệ!")
        
        print(f"Data shape: {data.shape}")
        print(f"CRS gốc: {src_crs}")
        print(f"Extent WGS84: {extent_wgs84}")

        # Tạo figure
        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
        fig.patch.set_facecolor('white')

        print("Vẽ bản đồ...")
        
        # Colormap cho phân ngưỡng
        cmap_threshold, label_threshold = get_threshold_colormap()

        # Kiểm tra kích thước ảnh
        total_pixels = data.shape[0] * data.shape[1]
        use_basemap = total_pixels < 100_000_000
        
        if use_basemap:
            try:
                print("Đang tải bản đồ nền...")
                ax.set_xlim(extent_original[0], extent_original[1])
                ax.set_ylim(extent_original[2], extent_original[3])
                ctx.add_basemap(ax, crs=src_crs.to_string(),
                              source=ctx.providers.OpenStreetMap.Mapnik,
                              alpha=0.5, zorder=1, attribution=False)
                print("✓ Đã thêm bản đồ nền")
            except Exception as e:
                print(f"Không thể tải basemap: {e}")
        else:
            print(f"⚠ Bỏ qua basemap do ảnh quá lớn ({total_pixels:,} pixels)")

        # Vẽ dữ liệu phân ngưỡng
        im = ax.imshow(data_masked, extent=extent_original,
                      cmap=cmap_threshold, vmin=0, vmax=5,
                      aspect='auto', alpha=0.85, zorder=3, interpolation='nearest')

        # Colorbar với chú giải
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, 
                           boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
                           ticks=[1, 2, 3, 4, 5])
        cbar.set_label(label_threshold, fontsize=12, fontweight='bold')
        
        # Thiết lập labels cho 5 mức với giá trị ngưỡng - căn giữa mỗi màu
        tick_labels = [
            'Rất thấp\n(0.000-0.125)',
            'Thấp\n(0.126-0.282)',
            'Trung bình\n(0.283-0.475)',
            'Cao\n(0.476-0.741)',
            'Rất cao\n(0.742-1.000)'
        ]
        cbar.ax.set_yticklabels(tick_labels, fontsize=9)
        cbar.ax.tick_params(labelsize=9)

        # Grid tọa độ
        lon_range = extent_wgs84[1] - extent_wgs84[0]
        lat_range = extent_wgs84[3] - extent_wgs84[2]
        width_range = extent_original[1] - extent_original[0]
        height_range = extent_original[3] - extent_original[2]

        lon_ticks_wgs = np.linspace(extent_wgs84[0], extent_wgs84[1], 8)
        lat_ticks_wgs = np.linspace(extent_wgs84[2], extent_wgs84[3], 6)
        
        transformer_inv = Transformer.from_crs('EPSG:4326', src_crs, always_xy=True) if src_crs and src_crs.to_string() != 'EPSG:4326' else None
        if transformer_inv:
            lon_ticks_orig = [transformer_inv.transform(lon, extent_wgs84[2])[0] for lon in lon_ticks_wgs]
            lat_ticks_orig = [transformer_inv.transform(extent_wgs84[0], lat)[1] for lat in lat_ticks_wgs]
        else:
            lon_ticks_orig = lon_ticks_wgs
            lat_ticks_orig = lat_ticks_wgs

        ax.set_xticks(lon_ticks_orig)
        ax.set_yticks(lat_ticks_orig)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='black', zorder=3)

        # Format nhãn
        lon_labels = [f'{lon:.3f}°Đ' for lon in lon_ticks_wgs]
        lat_labels = [f'{lat:.3f}°B' for lat in lat_ticks_wgs]
            
        ax.set_xticklabels(lon_labels, fontsize=9)
        ax.set_yticklabels(lat_labels, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=9, top=True, right=True, labeltop=True, labelright=True)

        # VẼ LA BÀN CẢI TIẾN - căn chỉnh vị trí
        print("Vẽ la bàn...")
        draw_compass_rose(ax, x_pos=0.93, y_pos=0.90, size=0.040)

        # Scale bar
        km_per_deg = 111.32 * np.cos(np.radians((extent_wgs84[2]+extent_wgs84[3])/2))

        x_start = extent_original[0] + width_range * 0.05
        y_pos = extent_original[2] + height_range * 0.08

        segments = [0, 10, 25, 50]  # km
        bar_colors = ['black', 'white', 'black']
        
        scale_factor = width_range / lon_range
        
        # Nền trắng cho scale bar
        total_width = (segments[-1] / km_per_deg) * scale_factor
        bg_rect = Rectangle((x_start, y_pos - height_range*0.025),
                           total_width * 1.3, height_range*0.045,
                           facecolor='white', edgecolor='black', linewidth=1.5, zorder=9, alpha=0.9)
        ax.add_patch(bg_rect)

        # Vẽ scale bar
        for i in range(len(segments)-1):
            width_km = segments[i+1] - segments[i]
            width = (width_km / km_per_deg) * scale_factor
            rect = Rectangle((x_start + (segments[i]/km_per_deg) * scale_factor, y_pos),
                            width, height_range*0.01,
                            facecolor=bar_colors[i], edgecolor='black', linewidth=1, zorder=10)
            ax.add_patch(rect)

        # Labels cho scale bar
        for km in segments:
            ax.text(x_start + (km/km_per_deg) * scale_factor,
                   y_pos - height_range*0.005, str(km),
                   ha='center', va='top', fontsize=10, fontweight='bold', zorder=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

        ax.text(x_start + (segments[-1]/km_per_deg) * scale_factor * 1.15,
               y_pos + height_range*0.005, 'km',
               ha='left', va='center', fontsize=11, fontweight='bold', zorder=11)

        # Attribution
        fig.text(0.99, 0.01, '© OpenStreetMap contributors | Machine Learning Model',
                ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)

        # Labels
        ax.set_xlabel('Kinh độ', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vĩ độ', fontsize=12, fontweight='bold')

        # Tiêu đề - trích xuất từ tên file
        base_name = filename.replace('cliped_', '').replace('_thresholded', '').replace('.tif', '').replace('.tiff', '')
        parts = base_name.replace('flood_susceptibility_', '').replace('flood_probability_', '').split('_')
        if len(parts) >= 2:
            algorithm = parts[0].upper()  # PSO, PUMA, RSO
            model = parts[1].upper()      # RF, SVR, XGB
            title = f'BẢN ĐỒ MỨC ĐỘ NHẠY CẢM VỚI NGẬP LỤT - {algorithm} + {model}'
        else:
            title = 'BẢN ĐỒ MỨC ĐỘ NHẠY CẢM VỚI NGẬP LỤT'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout(rect=[0, 0.02, 1, 1])

        print(f"Lưu file: {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Đã tạo bản đồ thành công!")
        
    except Exception as e:
        print(f"✗ LỖI khi xử lý: {str(e)}")
        import traceback
        traceback.print_exc()

# ===== TỔNG KẾT =====
print("\n" + "="*80)
print("✓ HOÀN THÀNH TẤT CẢ!")
print("="*80)
print(f"Kết quả lưu tại: {output_folder}")
print("="*80)