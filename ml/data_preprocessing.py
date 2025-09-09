import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from config import DEFAULT_FLOOD_CONFIG

def debug_flood_data(config=None):
    """
    Debug dữ liệu để kiểm tra phân bố và thứ tự
    """
    if config is None:
        config = DEFAULT_FLOOD_CONFIG
    
    # Đọc dữ liệu
    df = pd.read_csv(
        config['file_path'], 
        sep=config['separator'], 
        na_values=config['na_values']
    )
    
    label_column = config['label_column']
    
    print("=== DEBUG THÔNG TIN DỮ LIỆU ===")
    print(f"Kích thước DataFrame: {df.shape}")
    print(f"Tên cột nhãn: {label_column}")
    
    if label_column in df.columns:

        # Kiểm tra xem có bị sort theo nhãn không
        is_sorted_desc = df[label_column].is_monotonic_decreasing
        is_sorted_asc = df[label_column].is_monotonic_increasing
        print(f"Dữ liệu được sort giảm dần: {is_sorted_desc}")
        print(f"Dữ liệu được sort tăng dần: {is_sorted_asc}")


"""
def detect_and_normalize_features(X, feature_columns):
    '''
    Phát hiện và chuẩn hóa các cột chưa được chuẩn hóa
    
    Args:
        X: numpy array của features
        feature_columns: danh sách tên các cột
    
    Returns:
        X_normalized: numpy array đã được chuẩn hóa
        normalized_cols: danh sách các cột đã được chuẩn hóa
    '''
    X_normalized = X.copy()
    normalized_cols = []
    
    # Kiểm tra từng cột
    for i, col in enumerate(feature_columns):
        col_data = X[:, i]
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        
        # Kiểm tra nếu cột chưa chuẩn hóa (giá trị không nằm trong khoảng [0,1])
        if min_val < 0 or max_val > 1:
            # Thực hiện chuẩn hóa Min-Max
            X_normalized[:, i] = (col_data - min_val) / (max_val - min_val)
            normalized_cols.append(col)
            
    return X_normalized, normalized_cols
"""

def prepare_flood_data(config=None, shuffle_data=True, debug=False):
    """
    Chuẩn bị dữ liệu lũ lụt với tùy chọn shuffle để tránh data leakage
    
    Args:
        config: Cấu hình dữ liệu
        shuffle_data: Có shuffle dữ liệu hay không (khuyến nghị: True)
        debug: Có debug thông tin hay không
    """
    if config is None:
        config = DEFAULT_FLOOD_CONFIG
    
    # Debug trước khi xử lý
    if debug:
        debug_flood_data(config)
    
    # Đọc dữ liệu
    df = pd.read_csv(
        config['file_path'], 
        sep=config['separator'], 
        na_values=config['na_values']
    )
    
    feature_columns = config['feature_columns']
    label_column = config['label_column']
    
    # Kiểm tra và convert nhãn nếu cần
    if df[label_column].dtype == 'object':
        # Nếu là string, thử convert
        if 'Yes' in df[label_column].values or 'No' in df[label_column].values:
            df[label_column] = (df[label_column] == 'Yes').astype(float)
        else:
            # Thử convert thành numeric
            df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
    
    # Đảm bảo nhãn là 0 hoặc 1
    df[label_column] = df[label_column].astype(float)
    
    # QUAN TRỌNG: Shuffle dữ liệu để tránh ordering bias
    if shuffle_data:
        df = shuffle(df, random_state=config.get('random_state', 42))
        print(f"✅ Đã shuffle dữ liệu với random_state={config.get('random_state', 42)}")
    
    # Kiểm tra phân bố sau khi xử lý
    print(f"\n=== PHÂN BỐ NHÃN SAU XỬ LÝ ===")
    value_counts = df[label_column].value_counts().sort_index()
    print(value_counts)
    print(f"Tỷ lệ positive (flood=1): {df[label_column].mean():.3f}")
    
    # Cảnh báo nếu dữ liệu không cân bằng
    if df[label_column].mean() < 0.1 or df[label_column].mean() > 0.9:
        print("⚠️ CẢNH BÁO: Dữ liệu rất không cân bằng!")
    
    # Chuyển đổi dấu phẩy thành dấu chấm cho cột numeric
    for col in feature_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Chuẩn bị X và y
    X = df[feature_columns].values
    y = np.array(df[label_column].values)
    
    # Xử lý missing values nếu có
    if np.isnan(X).any():
        print("⚠️ Phát hiện missing values trong features, đang impute...")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # Kiểm tra cuối cùng
    print(f"\n=== KIỂM TRA CUỐI CÙNG ===")
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    print(f"Y unique values: {np.unique(y, return_counts=True)}")
    
    return X, y, feature_columns

def get_feature_info(config=None):
    """Lấy thông tin về đặc trưng"""
    if config is None:
        config = DEFAULT_FLOOD_CONFIG
    
    return config['feature_columns'], config['label_column']