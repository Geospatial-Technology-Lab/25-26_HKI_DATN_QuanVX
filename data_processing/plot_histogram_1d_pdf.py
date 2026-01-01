"""
Script vẽ histogram 1D và Probability Density Function (PDF)
cho giá trị dự đoán của các mô hình RF, SVM, XGB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def plot_histogram_1d_pdf(data_dict, output_file, title="Probability Density Function"):
    """
    Vẽ histogram 1D và PDF cho nhiều mô hình
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary với key là tên mô hình và value là array giá trị prediction
    output_file : str
        Đường dẫn file output PNG
    title : str
        Tiêu đề biểu đồ
    """
    # Tạo figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('white')
    
    # Định nghĩa màu sắc cho từng mô hình
    colors = {
        'PSO_RF': '#1f77b4',
        'PSO_SVM': '#ff7f0e', 
        'PSO_XGB': '#2ca02c',
        'PUMA_RF': '#d62728',
        'PUMA_SVM': '#9467bd',
        'PUMA_XGB': '#8c564b',
        'RS_RF': '#e377c2',
        'RS_SVM': '#7f7f7f',
        'RS_XGB': '#bcbd22'
    }
    
    # ===== SUBPLOT 1: HISTOGRAM =====
    for model_name, predictions in data_dict.items():
        color = colors.get(model_name, '#000000')
        
        # Vẽ histogram
        ax1.hist(predictions, bins=50, alpha=0.5, label=model_name, 
                color=color, edgecolor='black', linewidth=0.5, density=True)
    
    ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Histogram - Predicted Values Distribution', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    ax1.set_xlim(0, 1)
    
    # ===== SUBPLOT 2: PDF (Kernel Density Estimation) =====
    x_range = np.linspace(0, 1, 1000)
    
    for model_name, predictions in data_dict.items():
        color = colors.get(model_name, '#000000')
        
        # Tính KDE (Kernel Density Estimation)
        try:
            kde = gaussian_kde(predictions, bw_method='scott')
            pdf = kde(x_range)
            
            # Vẽ PDF
            ax2.plot(x_range, pdf, label=model_name, color=color, linewidth=2.5, alpha=0.8)
            
            # Fill area under curve
            ax2.fill_between(x_range, pdf, alpha=0.15, color=color)
            
        except Exception as e:
            print(f"  ⚠ Không thể tính KDE cho {model_name}: {e}")
    
    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Density Function (PDF)', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    
    # Thêm tiêu đề chung
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Căn chỉnh layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Lưu file
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Đã lưu biểu đồ: {output_file}")
    plt.close()


def plot_individual_pdf(predictions, model_name, output_file):
    """
    Vẽ PDF riêng cho từng mô hình
    
    Parameters:
    -----------
    predictions : array-like
        Mảng giá trị dự đoán
    model_name : str
        Tên mô hình
    output_file : str
        Đường dẫn file output
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('white')
    
    # Histogram
    n, bins, patches = ax.hist(predictions, bins=50, alpha=0.6, 
                                color='steelblue', edgecolor='black', 
                                linewidth=0.8, density=True, label='Histogram')
    
    # KDE (PDF)
    x_range = np.linspace(predictions.min(), predictions.max(), 1000)
    kde = gaussian_kde(predictions, bw_method='scott')
    pdf = kde(x_range)
    
    ax.plot(x_range, pdf, 'r-', linewidth=2.5, label='PDF (KDE)', alpha=0.9)
    ax.fill_between(x_range, pdf, alpha=0.2, color='red')
    
    # Thống kê
    mean_val = np.mean(predictions)
    median_val = np.median(predictions)
    std_val = np.std(predictions)
    
    # Vẽ đường mean và median
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_val:.4f}', alpha=0.8)
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
               label=f'Median = {median_val:.4f}', alpha=0.8)
    
    # Thiết lập
    ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title(f'PDF - {model_name}\n(μ={mean_val:.4f}, σ={std_val:.4f})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """
    Hàm chính
    """
    # Cấu hình đường dẫn
    input_dir = r"D:\prj\results\validate"
    output_dir = r"D:\prj\results\histograms"
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách các mô hình
    models = {
        'PSO_RF': 'validation_pso_RF.csv',
        'PSO_SVM': 'validation_pso_SVM.csv',
        'PSO_XGB': 'validation_pso_XGB.csv',
        'PUMA_RF': 'validation_puma_RF.csv',
        'PUMA_SVM': 'validation_puma_SVM.csv',
        'PUMA_XGB': 'validation_puma_XGB.csv',
        'RS_RF': 'validation_rs_RF.csv',
        'RS_SVM': 'validation_rs_SVM.csv',
        'RS_XGB': 'validation_rs_XGB.csv'
    }
    
    print("="*80)
    print("CHƯƠNG TRÌNH VẼ HISTOGRAM 1D VÀ PDF")
    print("="*80)
    print(f"Thư mục input: {input_dir}")
    print(f"Thư mục output: {output_dir}")
    print("="*80)
    
    # Dictionary lưu dữ liệu tất cả mô hình
    all_predictions = {}
    
    # Đọc dữ liệu từ CSV
    for model_name, csv_filename in models.items():
        csv_file = os.path.join(input_dir, csv_filename)
        
        if not os.path.exists(csv_file):
            print(f"\n⚠ Không tìm thấy file: {csv_file}")
            continue
        
        print(f"\nĐọc dữ liệu: {model_name}")
        df = pd.read_csv(csv_file)
        
        if 'prediction' not in df.columns:
            print(f"  ✗ Lỗi: Không có cột 'prediction'")
            continue
        
        predictions = df['prediction'].values
        predictions = predictions[~np.isnan(predictions)]
        
        all_predictions[model_name] = predictions
        
        print(f"  - Số điểm: {len(predictions):,}")
        print(f"  - Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
        print(f"  - Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
        
        # Vẽ PDF riêng cho từng mô hình
        individual_output = os.path.join(output_dir, f'pdf_individual_{model_name}.png')
        plot_individual_pdf(predictions, model_name, individual_output)
        print(f"  ✓ Đã lưu PDF riêng: pdf_individual_{model_name}.png")
    
    # Vẽ biểu đồ tổng hợp tất cả mô hình
    if all_predictions:
        print("\n" + "="*80)
        print("Vẽ biểu đồ tổng hợp...")
        
        combined_output = os.path.join(output_dir, 'pdf_combined_all_models.png')
        plot_histogram_1d_pdf(all_predictions, combined_output, 
                            "Prediction Distribution - All Models")
        
        print("="*80)
        print("✓ HOÀN THÀNH!")
        print("="*80)
    else:
        print("\n✗ Không có dữ liệu để vẽ!")


if __name__ == "__main__":
    main()
