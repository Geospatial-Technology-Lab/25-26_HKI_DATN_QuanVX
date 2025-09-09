from unittest import result
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from typing import Dict, Any, Tuple, Optional, Union
from puma_optimizer import PUMAOptimizer, RANDOM_SEED
from model_params import get_param_ranges, OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model
from data_preprocessing import prepare_flood_data, get_feature_info
warnings.filterwarnings('ignore')

class RandomForestPUMAOptimizer:
    """Bộ tối ưu hóa PUMA cho Random Forest Regressor."""
    
    def __init__(self, X: Union[np.ndarray, list] = None, y: Union[np.ndarray, list] = None, 
                 population_size: Optional[int] = None, generations: Optional[int] = None):
        """Khởi tạo bộ tối ưu hóa PUMA cho Random Forest Regressor."""
        # Sử dụng cấu hình mặc định nếu không được cung cấp
        if population_size is None:
            population_size = OPTIMIZATION_CONFIG['population_size']
        if generations is None:
            generations = OPTIMIZATION_CONFIG['generations']
            
        # Khởi tạo PUMA optimizer - TẬN DỤNG CẤU TRÚC CŨ
        self.optimizer = PUMAOptimizer(
            X=X, y=y,  # Có thể truyền dữ liệu hoặc để None để dùng mặc định
            model_type='regression', 
            population_size=population_size, 
            generations=generations,
            random_state=RANDOM_SEED
        )
        
        # Lấy phạm vi tham số RF từ file cấu hình
        param_ranges = get_param_ranges('rf')
        
        self.optimizer.set_param_ranges(param_ranges)
        self.optimizer.set_evaluate_function(self.evaluate_rf)
    
    def _create_rf_model(self, params: Dict[str, Any]) -> RandomForestRegressor:
        """Tạo RandomForestRegressor từ tham số - TẬN DỤNG LOGIC CŨ."""
        # Chỉ lấy các tham số hợp lệ cho RandomForestRegressor
        valid_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', None),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
            'max_features': params.get('max_features', 'sqrt'),
            'bootstrap': params.get('bootstrap', True),
            'n_jobs': -1,
            'random_state': RANDOM_SEED
        }
        
        # Thêm các tham số tùy chọn nếu có
        optional_params = ['max_leaf_nodes', 'min_impurity_decrease']
        for param in optional_params:
            if param in params:
                valid_params[param] = params[param]
        
        return RandomForestRegressor(**valid_params)
    
    def evaluate_rf(self, individual: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> float:
        """Đánh giá fitness của Random Forest với các tham số - TẬN DỤNG HÀM CŨ."""
        # Tạo mô hình Random Forest Regressor
        model = self._create_rf_model(individual)

        # Sử dụng hàm đánh giá chung từ evaluation_utils - TẬN DỤNG HÀM CŨ
        return evaluate_regression_model(
            model, X_train, X_test, y_train, y_test, 
            clip_predictions=True, return_detailed=False
        )
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        return self.optimizer.optimize(verbose=True, save_csv=True, save_model=True)
    
    def get_best_model(self) -> RandomForestRegressor:
        """Lấy mô hình tốt nhất sau khi tối ưu hóa."""
        if self.optimizer.best_individual is None:
            raise ValueError("Chưa chạy tối ưu hóa! Hãy gọi optimize() trước.")
        
        return self._create_rf_model(self.optimizer.best_individual)

def main() -> None:
    """Hàm chính để chạy quá trình tối ưu hóa Random Forest."""
    try:
        # Sử dụng module data_preprocessing để chuẩn bị dữ liệu
        print("Bắt đầu chuẩn bị dữ liệu...")
        X, y, _ = prepare_flood_data(shuffle_data=True, debug=False)
        
        # Lấy thông tin về đặc trưng
        feature_names, label_column = get_feature_info()
        print(f"Số lượng đặc trưng: {len(feature_names)}")
        print(f"Các đặc trưng: {feature_names}")
        
        # Khởi tạo và chạy bộ tối ưu PUMA cho RF
        print("Bắt đầu tối ưu hóa PUMA cho Random Forest...")
        rf_optimizer = RandomForestPUMAOptimizer()
        result = rf_optimizer.optimize()

        # Lấy kết quả từ dictionary trả về
        best_params = result['best_params']
        best_score = result['best_score']
        best_model = result['best_model']
        
        # In kết quả cuối cùng
        print("\n=== Kết quả cuối cùng ===")
        print(f"Điểm số tổng hợp tốt nhất: {best_score:.4f}")
        print("\nTham số tối ưu:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        # Lấy mô hình cuối cùng
        final_model = result['best_model']

        # Huấn luyện và đánh giá trên tập kiểm tra
        final_model.fit(rf_optimizer.optimizer.X_train_scaled, rf_optimizer.optimizer.y_train)
        y_pred = final_model.predict(rf_optimizer.optimizer.X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Giới hạn dự đoán từ 0 đến 1
        
        # Tính toán và lưu metrics cuối cùng
        y_test = np.array(rf_optimizer.optimizer.y_test)
        final_r2 = r2_score(y_test, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        final_mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nMetrics cuối cùng:")
        print(f"R²: {final_r2:.4f}")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        
        # Lưu kết quả tối ưu hóa
        print(f"File CSV kết quả: {result['csv_file']}")
        print(f"File mô hình tốt nhất: {result['model_file']}")
        
        print(f"\n{'='*60}")
        print("HOÀN THÀNH TỐI ƯU HÓA RANDOM FOREST")
            
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file dữ liệu! {e}")
        print("Vui lòng kiểm tra đường dẫn dataset.")
    except pd.errors.EmptyDataError:
        print("Lỗi: File dữ liệu trống!")
    except pd.errors.ParserError as e:
        print(f"Lỗi: Không thể đọc file CSV! {e}")
    except ValueError as e:
        print(f"Lỗi giá trị: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()