import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS
import warnings
warnings.filterwarnings('ignore')

class FloodDataPreprocessor:
    def __init__(self):

        self.feature_columns = FEATURE_COLUMNS
        
    def load_data(self, file_path):

        try:
            data = pd.read_csv(file_path)
            print(f"📊 Đã đọc dữ liệu: {data.shape[0]:,} điểm, {data.shape[1]} cột")
            
            # Kiểm tra tọa độ
            coords = data.iloc[:, :2]
            print(f"🌍 Phạm vi tọa độ:")
            print(f"   X: {coords.iloc[:, 0].min():.4f}° - {coords.iloc[:, 0].max():.4f}°")
            print(f"   Y: {coords.iloc[:, 1].min():.4f}° - {coords.iloc[:, 1].max():.4f}°")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {file_path}")
            raise
        except Exception as e:
            print(f"❌ Lỗi đọc dữ liệu: {e}")
            raise
    
    def validate_features(self, data):
        """Kiểm tra và validate features"""
        missing_features = []
        
        for feature in self.feature_columns:
            if feature not in data.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(f"⚠️ Thiếu features: {missing_features}")
            print(f"📋 Có sẵn: {list(data.columns)}")
            raise ValueError(f"Thiếu features: {missing_features}")
        
        # Kiểm tra missing values
        missing_counts = data[self.feature_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print("⚠️ Missing values:")
            for feature, count in missing_counts[missing_counts > 0].items():
                print(f"   {feature}: {count}")
        
        print(f"✅ Tất cả {len(self.feature_columns)} features đều có sẵn")
        
    def process(self, data):

        # Validate features
        self.validate_features(data)
        
        # Lấy features (13 cột)
        X = data[self.feature_columns].values
        
        # Lấy tọa độ (2 cột đầu tiên)
        coord_cols = data.columns[:2].tolist()
        coordinates = data[coord_cols]
        coordinates.columns = ['x', 'y']  # Chuẩn hóa tên cột
        
        print(f"🎯 Features shape: {X.shape}")
        print(f"📍 Coordinates shape: {coordinates.shape}")
        print(f"📋 Coordinate columns: {coord_cols}")
        
        # Kiểm tra NaN trong features
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"⚠️ Có {nan_count} giá trị NaN trong features")
            # Fill NaN bằng median
            X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
            print("✅ Đã fill NaN bằng median")
        
        return X, coordinates

def preprocess_data(file_path):

    print("🔧 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 50)
    
    preprocessor = FloodDataPreprocessor()
    
    # Đọc và xử lý dữ liệu
    data = preprocessor.load_data(file_path)
    X, coordinates = preprocessor.process(data)
    
    print("✅ HOÀN THÀNH TIỀN XỬ LÝ")
    print("=" * 50)
    
    return X, coordinates