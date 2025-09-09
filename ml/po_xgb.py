import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Dict, Any, Tuple, Optional, Union
from puma_optimizer import PUMAOptimizer, RANDOM_SEED
from model_params import get_param_ranges, OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model
from data_preprocessing import prepare_flood_data, get_feature_info
warnings.filterwarnings('ignore')

class XGBoostPUMAOptimizer:
    """Bộ tối ưu hóa PUMA cho XGBoost Regressor."""
    
    def __init__(self, X: Union[np.ndarray, list] = None, y: Union[np.ndarray, list] = None, 
                 population_size: Optional[int] = None, generations: Optional[int] = None):
        """Khởi tạo bộ tối ưu hóa PUMA cho XGBoost Regressor."""
        # Sử dụng cấu hình mặc định nếu không được cung cấp
        if population_size is None:
            population_size = OPTIMIZATION_CONFIG['population_size']
        if generations is None:
            generations = OPTIMIZATION_CONFIG['generations']
            
        # Khởi tạo PUMA optimizer - TÁN DỤNG CẤU TRÚC CŨ
        self.optimizer = PUMAOptimizer(
            X=X, y=y,  # Có thể truyền dữ liệu hoặc để None để dùng mặc định
            model_type='regression', 
            population_size=population_size, 
            generations=generations,
            random_state=RANDOM_SEED
        )
        
        # Lấy phạm vi tham số XGBoost từ file cấu hình
        param_ranges = get_param_ranges('xgb')
        
        self.optimizer.set_param_ranges(param_ranges)
        self.optimizer.set_evaluate_function(self.evaluate_xgboost)
    
    def _create_xgb_model(self, params: Dict[str, Any]) -> xgb.XGBRegressor:
        """Tạo XGBRegressor từ tham số - TÁN DỤNG LOGIC CŨ."""
        # Chỉ lấy các tham số hợp lệ cho XGBRegressor
        valid_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'subsample': params.get('subsample', 1.0),
            'colsample_bytree': params.get('colsample_bytree', 1.0),
            'colsample_bylevel': params.get('colsample_bylevel', 1.0),
            'colsample_bynode': params.get('colsample_bynode', 1.0),
            'reg_alpha': params.get('reg_alpha', 0.0),
            'reg_lambda': params.get('reg_lambda', 1.0),
            'min_child_weight': params.get('min_child_weight', 1),
            'gamma': params.get('gamma', 0.0),
            'max_delta_step': params.get('max_delta_step', 0),
            'scale_pos_weight': params.get('scale_pos_weight', 1.0),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        return xgb.XGBRegressor(**valid_params)
    
    def evaluate_xgboost(self, individual: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray) -> float:
        """Đánh giá fitness của XGBoost với các tham số - TÁN DỤNG HÀM CŨ."""
        # Tạo mô hình XGBoost Regressor
        model = self._create_xgb_model(individual)

        # Sử dụng hàm đánh giá chung từ evaluation_utils - TÁN DỤNG HÀM CŨ
        return evaluate_regression_model(
            model, X_train, X_test, y_train, y_test, 
            clip_predictions=True, return_detailed=False
        )
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        return self.optimizer.optimize(verbose=True, save_csv=True, save_model=True)
    
    def get_best_model(self) -> xgb.XGBRegressor:
        """Lấy mô hình tốt nhất sau khi tối ưu hóa."""
        if self.optimizer.best_individual is None:
            raise ValueError("Chưa chạy tối ưu hóa! Hãy gọi optimize() trước.")
        
        return self._create_xgb_model(self.optimizer.best_individual)

def main() -> None:
    """Hàm chính để chạy quá trình tối ưu hóa XGBoost."""
    try:
        # Sử dụng module data_preprocessing để chuẩn bị dữ liệu
        print("Bắt đầu chuẩn bị dữ liệu...")
        X, y, _ = prepare_flood_data(shuffle_data=True, debug=False)
        
        # Lấy thông tin về đặc trưng
        feature_names, label_column = get_feature_info()
        print(f"Số lượng đặc trưng: {len(feature_names)}")
        print(f"Các đặc trưng: {feature_names}")
        
        # Khởi tạo và chạy bộ tối ưu PUMA cho XGBoost
        print("Bắt đầu tối ưu hóa PUMA cho XGBoost...")
        xgb_optimizer = XGBoostPUMAOptimizer()
        result = xgb_optimizer.optimize()

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
        final_model.fit(xgb_optimizer.optimizer.X_train_scaled, xgb_optimizer.optimizer.y_train)
        y_pred = final_model.predict(xgb_optimizer.optimizer.X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Giới hạn dự đoán từ 0 đến 1
        
        # Tính toán và lưu metrics cuối cùng
        y_test = np.array(xgb_optimizer.optimizer.y_test)
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
        print("HOÀN THÀNH TỐI ƯU HÓA XGBOOST")
            
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