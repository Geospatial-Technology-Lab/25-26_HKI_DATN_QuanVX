import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
file_path = r"D:\25-26_HKI_DATN_QuanVX\train\data\flood_data_with_13_features.csv"
df = pd.read_csv(file_path)

# Loại bỏ cột nhãn (flood)
df_features = df.drop(columns=['flood'])

# Định nghĩa tên cột chuẩn (tiếng Việt có dấu hoặc viết tắt ngắn gọn)
column_mapping = {
    'lulc': 'LULC',
    'Density_River': 'Mật độ sông',
    'Density_Road': 'Mật độ đường',
    'Distan2river': 'Khoảng cách sông',
    'Distan2road_met': 'Khoảng cách đường',
    'aspect': 'Hướng sườn',
    'curvature': 'Độ cong',
    'dem': 'DEM',
    'flowDir': 'Hướng dòng chảy',
    'slope': 'Độ dốc',
    'twi': 'TWI',
    'NDVI': 'NDVI',
    'rainfall': 'Lượng mưa'
}

# Đổi tên cột
df_features = df_features.rename(columns=column_mapping)

# Tính ma trận tương quan
corr_matrix = df_features.corr()

# Tạo mask cho nửa trên (chỉ hiển thị nửa dưới)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Thiết lập kích thước figure
fig, ax = plt.subplots(figsize=(14, 12))

# Tạo ma trận annotation với dấu phẩy thập phân
annot_matrix = corr_matrix.applymap(lambda x: f'{x:.2f}'.replace('.', ','))

# Vẽ ma trận tương quan với nửa dưới
heatmap = sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=annot_matrix,  # Hiển thị giá trị với dấu phẩy
    fmt='',   # Không format vì đã format sẵn
    cmap='coolwarm',  # Bảng màu
    center=0,    # Tâm màu tại 0
    square=True, # Ô vuông
    linewidths=0.5,  # Độ rộng đường kẻ
    cbar_kws={"shrink": 0.8},  # Kích thước thanh màu
    vmin=-1,     # Giá trị min
    vmax=1,      # Giá trị max
    annot_kws={"size": 14},  # Tăng kích thước chữ số tương quan
    ax=ax
)

# Thay đổi định dạng colorbar sang dấu phẩy
cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'.replace('.', ',')))

# Thiết lập tiêu đề và labels
plt.xlabel('')
plt.ylabel('')

# Xoay labels trục x
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(rotation=0, fontsize=14)

# Căn chỉnh layout
plt.tight_layout()

# Lưu hình
output_path = r"D:\25-26_HKI_DATN_QuanVX\train\data\correlation_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Đã lưu ma trận tương quan tại: {output_path}")

# Hiển thị
plt.show()
