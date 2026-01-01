import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Thiết lập font hỗ trợ tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(results_dir):
    """
    Đọc dữ liệu từ thư mục results
    Returns: dictionary với key là (algorithm, model) và value là DataFrame
    """
    data = {}
    
    # Mapping tên file sang tên thuật toán và mô hình
    algorithms = {
        'pso': 'PSO',
        'puma': 'PUMA'
    }
    
    models = {
        'rf': 'RF',
        'svm': 'SVM',
        'xgb': 'XGB'
    }
    
    # Đọc tất cả file CSV trong thư mục
    for file in os.listdir(results_dir):
        if file.endswith('.csv'):
            # Parse tên file để lấy thuật toán và mô hình
            for alg_key, alg_name in algorithms.items():
                for model_key, model_name in models.items():
                    if alg_key in file.lower() and model_key in file.lower():
                        file_path = os.path.join(results_dir, file)
                        df = pd.read_csv(file_path)
                        data[(alg_name, model_name)] = df
                        break
    
    return data

def create_complete_overview(results_dir, output_file='complete_overview.png'):
    """
    Tạo biểu đồ tổng quan 3x3 cho 3 mô hình x 3 chỉ số
    """
    # Load dữ liệu
    data = load_data(results_dir)
    
    # Định nghĩa các mô hình và chỉ số
    models = ['RF', 'SVM', 'XGB']
    metrics = ['r2', 'mae', 'rmse']
    metric_names = {
        'r2': 'R²',
        'mae': 'MAE', 
        'rmse': 'RMSE'
    }
    
    # Màu sắc cho từng thuật toán
    colors = {
        'PSO': '#ff0000',     # Đỏ tươi
        'PUMA': '#00ff00'     # Xanh lá tươi
    }
    
    # Tạo figure với kích thước lớn
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    # Flag để vẽ chú giải chỉ một lần
    legend_added = False
    
    # Vẽ từng subplot
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            # Vẽ đường cho từng thuật toán
            for algorithm in ['PSO', 'PUMA']:
                key = (algorithm, model)
                if key in data:
                    df = data[key]
                    
                    # Xử lý cột iteration/generation
                    if 'iteration' in df.columns:
                        iterations = df['iteration'].values
                    elif 'generation' in df.columns:
                        iterations = df['generation'].values
                    else:
                        # Nếu không có cột nào, tạo index từ 1
                        iterations = np.arange(1, len(df) + 1)
                    
                    values = df[metric].values
                    
                    # Đảm bảo số vòng lặp từ 0-100
                    if len(iterations) > 100:
                        iterations = iterations[:100]
                        values = values[:100]
                    
                    ax.plot(iterations, values, 
                           color=colors[algorithm], 
                           linewidth=2, 
                           label=algorithm,
                           alpha=0.8)
            
            # Thiết lập tiêu đề cho subplot
            title = f'{model} - {metric_names[metric]}'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
            
            # Thiết lập trục X
            ax.set_xlim(0, 100)
            
            # Thiết lập trục Y dựa trên giá trị thực tế của từng metric
            # Tìm min/max từ dữ liệu thực tế
            y_values = []
            for alg in ['PSO', 'PUMA']:
                key = (alg, model)
                if key in data:
                    df = data[key]
                    y_values.extend(df[metric].values[:100])
            
            if y_values:
                y_min = min(y_values)
                y_max = max(y_values)
                # Thêm margin 10% cho đẹp
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Grid
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Nhãn trục chỉ hiển thị ở hàng dưới cùng và cột trái nhất
            if i == 2:  # Hàng cuối
                ax.set_xlabel('Số Vòng Lặp', fontsize=14)
            if j == 0:  # Cột đầu
                ax.set_ylabel(metric_names[metric], fontsize=14)
            
            # Thêm chú giải vào biểu đồ XGBoost-R² (góc trên bên phải)
            if i == 0 and j == 2 and not legend_added:
                ax.legend(loc='lower right', fontsize=14, framealpha=0.9,
                         edgecolor='gray', fancybox=True)
                legend_added = True
            
            # Định dạng tick
            ax.tick_params(labelsize=14)
            
            # Background màu nhẹ
            ax.set_facecolor('#f5f5f5')
    
    # Điều chỉnh layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Lưu file
    output_path = os.path.join(results_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Đã lưu biểu đồ tại: {output_path}')
    
    # Hiển thị
    plt.show()

if __name__ == '__main__':
    # Đường dẫn đến thư mục results
    results_dir = r'D:\25-26_HKI_DATN_QuanVX\results'
    
    # Tạo biểu đồ
    create_complete_overview(results_dir)
