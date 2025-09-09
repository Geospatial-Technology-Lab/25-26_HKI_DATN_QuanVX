"""
Tối ưu hóa Randomized Search cho XGBoost - Đã đồng bộ với rso_optimizer
"""

import numpy as np
import time
import warnings
from rso_optimizer import RandomizedSearch
from data_preprocessing import prepare_flood_data, get_feature_info
from config import FLOOD_DATA_CONFIG

warnings.filterwarnings('ignore')

def main():
    """Hàm chính để chạy quá trình tối ưu hóa XGBoost bằng Randomized Search."""
    try:
        # Chuẩn bị dữ liệu
        print("Bắt đầu chuẩn bị dữ liệu...")
        X, y, feature_columns = prepare_flood_data()
        
        # Lấy thông tin về đặc trưng
        feature_names, label_column = get_feature_info()
        print(f"Số lượng đặc trưng: {len(feature_names)}")
        print(f"Kích thước dữ liệu: {X.shape}")
        
        # Khởi tạo và chạy bộ tối ưu với Cross-Validation
        print("\nBắt đầu tối ưu hóa XGBoost bằng Randomized Search...")
        print("=" * 80)
        
        # Sử dụng RandomizedSearch từ rso_optimizer
        xgb_optimizer = RandomizedSearch(
            X=X, 
            y=y, 
            model_name='xgb', 
            model_type='regression',
            random_state=FLOOD_DATA_CONFIG.get('random_state', 42)
        )
        
        start_time = time.time()
        result = xgb_optimizer.search(
            n_iter=100,
            verbose=True,
            print_table=True,
            save_csv=True,
            save_model=True
        )
        end_time = time.time()
        
        print(f"\nThời gian tối ưu hóa: {end_time - start_time:.2f} giây")
        
        # In kết quả cuối cùng
        print("\n=== KẾT QUẢ CUỐI CÙNG ===")
        print(f"Điểm số tốt nhất: {result['best_score']:.6f}")
        print("\nTham số tối ưu:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
        
        if result['csv_file']:
            print(f"\nKết quả đã được lưu vào: {result['csv_file']}")
        if result['model_file']:
            print(f"Mô hình tốt nhất đã được lưu vào: {result['model_file']}")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()