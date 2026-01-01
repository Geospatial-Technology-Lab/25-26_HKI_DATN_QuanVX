import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Đọc dữ liệu từ file CSV
csv_file = r"D:\prj\results\dien_tich.csv"
df = pd.read_csv(csv_file, encoding='utf-8')

# Màu sắc cho 5 ngưỡng (giống file bieu_do_tron_nhieulop.py)
threshold_colors = ['#55FF00', '#AAFF00', '#FFFF00', '#FFAA00', '#F50000']
threshold_labels = ['Rất thấp', 'Thấp', 'Trung bình', 'Cao', 'Rất cao']

# Tên cột ngưỡng
thresholds = ['Ngưỡng 1 (km²)', 'Ngưỡng 2 (km²)', 'Ngưỡng 3 (km²)', 
              'Ngưỡng 4 (km²)', 'Ngưỡng 5 (km²)']

# Tổ chức dữ liệu theo cụm: RF, XGB, SVM
# Mỗi cụm có RS và PO
model_data = {
    'RF': {'RS': None, 'PO': None},
    'XGB': {'RS': None, 'PO': None},
    'SVM': {'RS': None, 'PO': None}
}

# Phân loại dữ liệu
for idx, row in df.iterrows():
    name = row['Tên ảnh']
    parts = name.split('/')
    model_type = parts[0].upper()
    if model_type == 'SVR':
        model_type = 'SVM'
    
    # Lấy thuật toán tối ưu
    optimization = parts[1].split('_')[-2].upper()
    if optimization == 'PUMA':
        optimization = 'PO'
    elif optimization == 'RSO':
        optimization = 'RS'
    
    # Lưu dữ liệu 5 ngưỡng
    if model_type in model_data and optimization in ['RS', 'PO']:
        model_data[model_type][optimization] = row[thresholds].values

# Tạo figure
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Thiết lập vị trí các cụm và cột
bar_width = 1.3  # Độ rộng mỗi cột (gấp đôi so với 0.35)
sub_gap = 0.1  # Khoảng cách giữa RS và PO trong cùng cụm
threshold_width = bar_width / 5  # Độ rộng của mỗi ngưỡng

# Tính toán vị trí để XGB ở giữa RF và SVM
# Tổng độ rộng của 1 cụm (RS + PO)
cluster_width = 2 * bar_width + sub_gap
# Khoảng cách giữa các cụm
spacing = 1.5
# Vị trí cụm: RF=0, XGB ở giữa, SVM cuối
cluster_positions = {
    'RF': 0, 
    'XGB': cluster_width + spacing,
    'SVM': 2 * (cluster_width + spacing)
}
clusters = ['RF', 'XGB', 'SVM']

# Vẽ các cột
x_ticks = []
x_labels = []

for cluster in clusters:
    base_pos = cluster_positions[cluster]
    
    # Vẽ RS
    if model_data[cluster]['RS'] is not None:
        rs_pos = base_pos
        data_rs = model_data[cluster]['RS']
        
        # Vẽ 5 cột cho 5 ngưỡng
        for i in range(5):
            x = rs_pos + i * threshold_width
            ax.bar(x, data_rs[i], threshold_width * 0.95, 
                   color=threshold_colors[i], edgecolor='black', 
                   linewidth=0.8, alpha=0.9)
        
        # Lưu vị trí nhãn (giữa 5 cột)
        x_ticks.append(rs_pos + 2 * threshold_width)
        x_labels.append(f'{cluster}\nRS')
    
    # Vẽ PO
    if model_data[cluster]['PO'] is not None:
        po_pos = base_pos + bar_width + sub_gap
        data_po = model_data[cluster]['PO']
        
        # Vẽ 5 cột cho 5 ngưỡng
        for i in range(5):
            x = po_pos + i * threshold_width
            ax.bar(x, data_po[i], threshold_width * 0.95, 
                   color=threshold_colors[i], edgecolor='black', 
                   linewidth=0.8, alpha=0.9)
        
        # Lưu vị trí nhãn
        x_ticks.append(po_pos + 2 * threshold_width)
        x_labels.append(f'{cluster}\nPO')

# Thiết lập trục x với khoảng trống bên phải cho chú giải
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, fontsize=14)

# Tính toán vị trí cuối cùng của SVM để thêm khoảng trống
svm_end = cluster_positions['SVM'] + bar_width + sub_gap + bar_width
legend_width = 2.5  # Độ rộng ước tính của ô chú giải
ax.set_xlim(-0.5, svm_end + legend_width)

# Thiết lập nhãn trục
ax.set_ylabel('Diện tích (km²)', fontsize=14, fontweight='bold')

# Thêm lưới ngang
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Định dạng trục y
ax.ticklabel_format(style='plain', axis='y')
ax.tick_params(axis='y', labelsize=14)

# Tạo chú giải cho các ngưỡng
legend_patches = []
for color, label in zip(threshold_colors, threshold_labels):
    patch = mpatches.Patch(facecolor=color, label=label, 
                          edgecolor='black', linewidth=0.8, alpha=0.9)
    legend_patches.append(patch)

# Hiển thị chú giải
legend = ax.legend(handles=legend_patches, 
                   loc='upper right',
                   fontsize=14,
                   title='Mức độ nhạy cảm',
                   title_fontsize=14,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   framealpha=0.95,
                   edgecolor='0.8',
                   ncol=1)

legend.get_title().set_weight('bold')

# Điều chỉnh layout
plt.tight_layout()

# Lưu hình
output_file = r"D:\prj\results\bieu_do_phan_bo_dien_tich.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Đã lưu biểu đồ tại: {output_file}")

# Hiển thị biểu đồ
plt.show()
