import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Đọc dữ liệu
df = pd.read_csv(r'D:\prj\results\dien_tich.csv')

# Thiết lập style
plt.style.use('seaborn-v0_8-pastel')
fig, ax = plt.subplots(figsize=(18, 14), subplot_kw=dict(aspect="equal"))
fig.patch.set_facecolor('white')

# Bảng màu
colors = ['#55FF00', '#AAFF00', '#FFFF00', '#FFAA00', '#F50000']
threshold_names = ['Rất thấp', 'Thấp', 
                   'Trung bình', 'Cao', 'Rất cao']

# Lấy tên mô hình và làm sạch
models = df['Tên ảnh'].values
model_labels = []
for model in models:
    # Trích xuất tên ngắn gọn
    parts = model.split('/')
    ml_model = parts[0].upper()  # rf, svm, xgb
    algorithm = parts[1].split('_')[2].upper()  # pso, puma, rso, po
    
    # Đổi PUMA thành PO
    if algorithm == 'PUMA':
        algorithm = 'PO'

    if algorithm == 'RSO':
        algorithm = 'RS'
    
    # Đổi tên mô hình ML
    if ml_model == 'RF':
        ml_name = 'RF'
    elif ml_model == 'SVR':
        ml_name = 'SVM'
    elif ml_model == 'XGB':
        ml_name = 'XGB'
    else:
        ml_name = ml_model
    
    model_labels.append(f"{algorithm} + {ml_name}")

# Thiết lập các vòng tròn đồng tâm
n_models = len(models)
inner_radius = 2.5
layer_width = 5
gap = 0.35

# Vẽ từng lớp
for i, model in enumerate(models):
    r_inner = inner_radius + i * (layer_width + gap)
    r_outer = r_inner + layer_width
    
    # Lấy dữ liệu 5 ngưỡng
    values = df.iloc[i, 1:6].values
    total = sum(values)
    percentages = (values / total) * 100
    
    # Bắt đầu từ 90 độ (12 giờ)
    start_angle = 90
    
    # Kiểm tra nếu là lớp PO + XGB (lớp thứ 2, index = 1)
    is_po_xgb = (model_labels[i] == "PO + XGB")
    
    for j, (value, pct, color) in enumerate(zip(values, percentages, colors)):
        angle = (value / total) * 360
        
        # Vẽ wedge với viền đặc biệt cho PO + XGB
        wedge = Wedge(
            center=(0, 0),
            r=r_outer,
            theta1=start_angle,
            theta2=start_angle + angle,
            width=layer_width,
            facecolor=color,
            edgecolor='#FF55FF' if is_po_xgb else 'white',  # Magenta sáng hơn cho PO + XGB
            linewidth=5 if is_po_xgb else 2.5,  # Viền dày hơn cho PO + XGB
            alpha=0.75
        )
        ax.add_patch(wedge)
        
        # Thêm % nếu phần đủ lớn (>25%)
        if pct > 25 or (pct < 22.4 and pct > 22.2):
            mid_angle = start_angle + angle / 2
            mid_radius = (r_inner + r_outer) / 2
            x = mid_radius * np.cos(np.radians(mid_angle))
            y = mid_radius * np.sin(np.radians(mid_angle))
            ax.text(x, y, f'{pct:.1f}%', 
                   ha='center', va='center',
                   fontsize=11, weight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='black', alpha=0.3, edgecolor='none'))
        
        start_angle += angle

# Tính max_radius trước khi vẽ nhãn
max_radius = inner_radius + n_models * (layer_width + gap) + 0.5

# Thêm nhãn mô hình với đường dẫn
for i, label in enumerate(model_labels):
    r_inner = inner_radius + i * (layer_width + gap)
    r_outer = r_inner + layer_width
    mid_radius = (r_inner + r_outer) / 2
    
    # Xác định vị trí nhãn (xen kẽ trái phải để tránh đè)
    if i % 2 == 0:  # Bên phải
        label_x = max_radius + 1
        # Phân bố đều theo chiều dọc
        label_y = -max_radius * 0.6 + (i * (max_radius * 1.2 / (n_models - 0.5)))
        ha = 'left'
        # Điểm kết nối trên vòng tròn (góc 0 độ - bên phải)
        connect_angle = 0
    else:  # Bên trái
        label_x = -(max_radius + 1)
        # Phân bố đều theo chiều dọc
        base_y = -max_radius * 0.6 + (i * (max_radius * 1.2 / (n_models - 1)))
        # Dịch PSO + SVR xuống 1 chút để tạo khoảng cách
        if i == 3:  # PSO + SVR
            label_y = base_y - 1.0
        else:
            label_y = base_y
        ha = 'right'
        # Điểm kết nối trên vòng tròn (góc 180 độ - bên trái)
        connect_angle = 180
    
    # Tính điểm kết nối trên vòng tròn
    connect_x = mid_radius * np.cos(np.radians(connect_angle))
    connect_y = mid_radius * np.sin(np.radians(connect_angle))
    
    # Vẽ đường dẫn từ vòng tròn đến nhãn
    middle_x = connect_x + (label_x - connect_x) * 0.5
    
    ax.plot([connect_x, middle_x, label_x - (0.15 if ha == 'left' else -0.15)], 
            [connect_y, label_y, label_y],
            color='#7F8C8D', linewidth=2, linestyle='-', alpha=0.5, zorder=1)
    
    # Vẽ điểm kết nối trên vòng tròn
    ax.plot(connect_x, connect_y, 'o', color='#34495E', markersize=5, zorder=5)
    
    # Vẽ nhãn
    ax.text(label_x, label_y, f'  {label}  ' if ha == 'left' else f'  {label}  ', 
           fontsize=12, weight='bold',
           ha=ha, va='center',
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#ECF0F1', edgecolor='#7F8C8D', 
                    linewidth=2, alpha=0.95),
           zorder=10)

# Vẽ vòng tròn trung tâm
center_circle = plt.Circle((0, 0), inner_radius, color='white', 
                           ec='#BDC3C7', linewidth=3, zorder=10)
ax.add_patch(center_circle)

# Thiết lập giới hạn
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)
ax.axis('off')

# Tạo chú giải đẹp
legend_elements = []
for color, name in zip(colors, threshold_names):
    legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='white', 
                                         linewidth=5, label=name, alpha=0.75))

legend = ax.legend(
    handles=legend_elements,
    loc='center left',
    bbox_to_anchor=(1.15, 0.5),
    fontsize=13,
    title='Mức độ nhạy cảm',
    title_fontsize=15,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='#BDC3C7',
    handleheight=2.0
)
legend.get_title().set_weight('bold')

# Lưu file
plt.savefig(r'D:\prj\results\bieu_do_tron_nhieulop.png', 
           dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Đã lưu biểu đồ tại: D:\\prj\\results\\bieu_do_tron_nhieulop.png")

plt.show()