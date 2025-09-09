"""
Tối ưu hóa PUMA cho MLP sử dụng bộ tối ưu hóa PUMA chung
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from typing import Dict, Any, Tuple, Optional, Union
from puma_optimizer import PUMAOptimizer, RANDOM_SEED
from model_params import get_param_ranges, OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model
from data_preprocessing import prepare_flood_data, get_feature_info
from config import get_output_path, get_timestamp_suffix, METRICS_CONFIG
warnings.filterwarnings('ignore')

class MLPPUMAOptimizer:
    """Bộ tối ưu hóa PUMA cho MLP Regressor."""
    
    def __init__(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], 
                 population_size: Optional[int] = None, generations: Optional[int] = None):
        """Khởi tạo bộ tối ưu hóa PUMA cho MLP Regressor."""
        # Sử dụng cấu hình từ OPTIMIZATION_CONFIG nếu không được cung cấp
        if population_size is None:
            population_size = OPTIMIZATION_CONFIG['population_size']
        if generations is None:
            generations = OPTIMIZATION_CONFIG['generations']
            
        self.optimizer = PUMAOptimizer(X, y, model_type='regression', 
                                     population_size=population_size, 
                                     generations=generations,
                                     random_state=OPTIMIZATION_CONFIG['random_seed'])
        
        # Lấy phạm vi tham số MLP từ file cấu hình
        param_ranges = get_param_ranges('mlp')
        
        self.optimizer.set_param_ranges(param_ranges)
        self.optimizer.set_evaluate_function(self.evaluate_mlp)
        
        # Ghi đè hàm calculate_metrics của optimizer để sử dụng MLP
        self.optimizer.calculate_metrics = self.calculate_mlp_metrics
    
    def calculate_mlp_metrics(self, individual: Dict[str, Any]) -> Tuple[float, float, float]:
        """Tính toán các metric chi tiết cho MLP regression"""
        try:
            # Tạo model MLP với tham số này
            model = self._create_mlp_model(individual)
            
            # Huấn luyện và dự đoán
            model.fit(self.optimizer.X_train_scaled, self.optimizer.y_train)
            y_pred = model.predict(self.optimizer.X_test_scaled)
            
            # Tính các metric
            r2 = r2_score(self.optimizer.y_test, y_pred)
            mae = mean_absolute_error(self.optimizer.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.optimizer.y_test, y_pred))
            
            return r2, mae, rmse
            
        except Exception as e:
            # Nếu có lỗi, trả về giá trị mặc định
            return 0.0, float('inf'), float('inf')
    
    def _create_mlp_model(self, params: Dict[str, Any]) -> MLPRegressor:
        """Tạo MLPRegressor từ tham số."""
        return MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            learning_rate=params.get('learning_rate', 'constant'),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            max_iter=params.get('max_iter', 200),
            beta_1=params.get('beta_1', 0.9),
            beta_2=params.get('beta_2', 0.999),
            epsilon=params.get('epsilon', 1e-8),
            validation_fraction=params.get('validation_fraction', 0.1),
            n_iter_no_change=params.get('n_iter_no_change', 10),
            tol=params.get('tol', 1e-4),
            random_state=RANDOM_SEED
        )
    
    def evaluate_mlp(self, individual: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, 
                    y_train: np.ndarray, y_test: np.ndarray) -> float:
        """Đánh giá fitness của MLP với các tham số."""
        # Tạo mô hình MLP Regressor sử dụng helper method
        model = self._create_mlp_model(individual)

        # Sử dụng hàm đánh giá chung cho regression (return_detailed=False để chỉ lấy fitness score)
        return evaluate_regression_model(model, X_train, X_test, y_train, y_test, 
                                       clip_predictions=True, return_detailed=False)
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """Chạy quá trình tối ưu hóa."""
        return self.optimizer.optimize(verbose=True)
    
    def get_best_model(self) -> MLPRegressor:
        """Lấy mô hình tốt nhất sau khi tối ưu hóa."""
        if not hasattr(self.optimizer, 'best_individual') or not self.optimizer.best_individual:
            raise ValueError("Chưa chạy tối ưu hóa! Hãy gọi optimize() trước.")
        
        return self._create_mlp_model(self.optimizer.best_individual)

def save_optimization_results(best_params: Dict[str, Any], best_score: float, 
                            detailed_metrics: Dict[str, float], 
                            output_prefix: str = 'po_mlp') -> None:
    """Lưu kết quả tối ưu hóa và metrics vào file."""
    timestamp = get_timestamp_suffix()
    
    # Lưu tham số tối ưu
    params_df = pd.DataFrame([
        {'Parameter': param, 'Value': str(value)} 
        for param, value in best_params.items()
    ])
    params_file = get_output_path(f'{output_prefix}_best_params_{timestamp}.csv')
    params_df.to_csv(params_file, index=False)
    
    # Lưu metrics chi tiết
    metrics_df = pd.DataFrame([
        {'Metric': 'Best Score', 'Value': best_score},
        {'Metric': 'R²', 'Value': detailed_metrics.get('r2', 0.0)},
        {'Metric': 'RMSE', 'Value': detailed_metrics.get('rmse', 0.0)},
        {'Metric': 'MAE', 'Value': detailed_metrics.get('mae', 0.0)},
        {'Metric': 'Fitness', 'Value': detailed_metrics.get('fitness', 0.0)}
    ])
    
    # Làm tròn theo cấu hình
    decimal_places = METRICS_CONFIG.get('decimal_places', 4)
    metrics_df['Value'] = metrics_df['Value'].round(decimal_places)
    
    metrics_file = get_output_path(f'{output_prefix}_metrics_{timestamp}.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"\nĐã lưu kết quả:")
    print(f"- Tham số tối ưu: {params_file}")
    print(f"- Metrics: {metrics_file}")

def calculate_and_display_final_metrics(model: MLPRegressor, optimizer: PUMAOptimizer, 
                                      model_name: str = 'MLP') -> Dict[str, float]:
    """Tính toán và hiển thị metrics cuối cùng sử dụng evaluate_regression_model."""
    # Sử dụng hàm đánh giá có sẵn với return_detailed=True
    detailed_results = evaluate_regression_model(
        model, 
        optimizer.X_train_scaled, 
        optimizer.X_test_scaled,
        optimizer.y_train, 
        optimizer.y_test, 
        clip_predictions=True, 
        return_detailed=True
    )
    
    print(f"\n=== Metrics cuối cùng cho {model_name} ===")
    print(f"R²: {detailed_results['r2']:.4f}")
    print(f"RMSE: {detailed_results['rmse']:.4f}")
    print(f"MAE: {detailed_results['mae']:.4f}")
    print(f"Fitness Score: {detailed_results['fitness']:.4f}")
    
    return detailed_results

def main() -> None:
    """Hàm chính để chạy quá trình tối ưu hóa MLP."""
    try:
        # Sử dụng module data_preprocessing đơn giản hóa để chuẩn bị dữ liệu
        print("Bắt đầu chuẩn bị dữ liệu...")
        X, y, feature_columns = prepare_flood_data()
        
        # Lấy thông tin về đặc trưng
        feature_names, label_column = get_feature_info()
        print(f"Số lượng đặc trưng: {len(feature_names)}")
        print(f"Các đặc trưng: {feature_names}")
        
        # Khởi tạo và chạy bộ tối ưu PUMA cho MLP
        print("Bắt đầu tối ưu hóa PUMA cho MLP...")
        mlp_optimizer = MLPPUMAOptimizer(X, y)
        best_params, best_score = mlp_optimizer.optimize()
        
        # In kết quả cuối cùng
        print("\n=== Kết quả tối ưu hóa PUMA ===")
        print(f"Điểm số tổng hợp tốt nhất: {best_score:.4f}")
        print("\nTham số tối ưu:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        # Lấy mô hình cuối cùng và đánh giá chi tiết
        final_model = mlp_optimizer.get_best_model()
        
        # Sử dụng hàm tiện ích để tính toán metrics chi tiết
        detailed_metrics = calculate_and_display_final_metrics(
            final_model, mlp_optimizer.optimizer, 'MLP'
        )
        
        # Lưu kết quả sử dụng hàm tiện ích
        save_optimization_results(best_params, best_score, detailed_metrics, 'po_mlp')
            
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file dữ liệu! {e}")
        print("Vui lòng kiểm tra đường dẫn dataset trong config.py")
    except pd.errors.EmptyDataError:
        print("Lỗi: File dữ liệu trống!")
    except pd.errors.ParserError as e:
        print(f"Lỗi: Không thể đọc file CSV! {e}")
        print("Kiểm tra định dạng file và separator trong FLOOD_DATA_CONFIG")
    except ValueError as e:
        print(f"Lỗi giá trị: {e}")
        print("Kiểm tra cấu hình tham số trong model_params.py")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        print("\nGợi ý: Kiểm tra các import và cấu hình trong config.py và model_params.py")

if __name__ == "__main__":
    main()
