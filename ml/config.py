import os
from pathlib import Path

# ========== ĐƯỜNG DẪN VÀ FILE ==========
DATA_CONFIG = {
    'flood_csv_path': r"/run/media/quan/Quan Vu/25-26_HKI_DATN_QuanVX/merged_flood_point_merge_cleaned_balanced_reordered_nonlatlon_normalized_delindexNB.csv",
    'output_dir': './outputs',
    'models_dir': './models',
    'results_dir': './results'
}

# ========== CẤU HÌNH DỮ LIỆU LŨ LỤT ==========
FLOOD_DATA_CONFIG = {
    'separator': ',',
    'na_values': '<Null>',
    'feature_columns': [
        'lulc', 'Density_River', 'Density_Road', 'Distan2river_met', 'Distan2road_met',
        'aspect', 'curvature', 'dem', 'flowDir', 'slope', 'twi', 
        'NDVI', 'rainfall'
    ],
    'label_column': 'flood',
    'target_mapping': {1: 1.0, 0: 0.0},
    'test_size': 0.2,
    'random_state': 42,
    'imputation_strategy': 'median'
}

# ========== CẤU HÌNH METRICS ==========
METRICS_CONFIG = {
    'regression_metrics': ['r2', 'mse', 'mae', 'rmse'],
    'classification_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'save_format': 'csv',
    'decimal_places': 4
}

# ========== CẤU HÌNH VISUALIZATION ==========
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_format': 'png',
    'style': 'seaborn-v0_8',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

# ========== HÀM TIỆN ÍCH ==========
def ensure_directories():
    """Đảm bảo các thư mục cần thiết tồn tại."""
    for dir_path in [DATA_CONFIG['output_dir'], DATA_CONFIG['models_dir'], DATA_CONFIG['results_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
def get_output_path(filename: str, subdir: str = 'output_dir') -> str:
    """Lấy đường dẫn file output với thư mục phù hợp."""
    ensure_directories()
    return os.path.join(DATA_CONFIG[subdir], filename)

def get_timestamp_suffix() -> str:
    """Lấy timestamp để tạo tên file duy nhất."""
    import time
    return str(int(time.time()))

# ========== CẤU HÌNH CHO DATA_PREPROCESSING ==========
DEFAULT_FLOOD_CONFIG = {
    'file_path': DATA_CONFIG['flood_csv_path'],
    'separator': FLOOD_DATA_CONFIG['separator'],
    'na_values': FLOOD_DATA_CONFIG['na_values'],
    'feature_columns': FLOOD_DATA_CONFIG['feature_columns'],
    'label_column': FLOOD_DATA_CONFIG['label_column'],
    'target_mapping': FLOOD_DATA_CONFIG['target_mapping']
}

# ========== XUẤT CẤU HÌNH ==========
__all__ = [
    'DATA_CONFIG',
    'FLOOD_DATA_CONFIG', 
    'METRICS_CONFIG',
    'PLOT_CONFIG',
    'DEFAULT_FLOOD_CONFIG',
    'ensure_directories',
    'get_output_path',
    'get_timestamp_suffix'
]