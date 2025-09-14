"""
Tối ưu hóa RSO cho MLP sử dụng bộ tối ưu hóa RSO chung
"""

import numpy as np
import time
import warnings
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.neural_network import MLPRegressor
from ml.rso_optimizer import RSOOptimizer, RANDOM_SEED
from model_params import get_param_ranges, OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model
from data_preprocessing import prepare_flood_data, get_feature_info
import random
warnings.filterwarnings('ignore')

class MLPRSOOptimizer:
    """Bộ tối ưu hóa RSO cho MLP Regressor."""
    
    def __init__(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], 
                 n_iterations: Optional[int] = None):
        """Khởi tạo bộ tối ưu hóa RSO cho MLP Regressor."""
        # Sử dụng cấu hình mặc định nếu không được cung cấp
        if n_iterations is None:
            n_iterations = OPTIMIZATION_CONFIG.get('generations', 100)
            
        self.optimizer = RSOOptimizer(X, y, model_type='regression', 
                                    n_iterations=n_iterations)
        
        # Lấy phạm vi tham số MLP từ file cấu hình
        param_ranges = get_param_ranges('mlp')
        
        self.optimizer.set_param_ranges(param_ranges)
        self.optimizer.set_evaluate_function(self.evaluate_mlp)
        
        # Thêm method để tạo model cho detailed metrics
        self.optimizer._create_model_for_evaluation = self._create_mlp_model
    
    def _create_mlp_model(self, params: Dict[str, Any]) -> MLPRegressor:
        """Tạo MLPRegressor từ tham số."""
        # Điều chỉnh tham số dựa trên solver
        mlp_params = params.copy()
        
        if params.get('solver') == 'lbfgs':
            # LBFGS chỉ hoạt động tốt với mạng nhỏ
            simple_layers = [(50,), (100,), (150,), (200,), (50, 50), (100, 50), (100, 100)]
            if len(params.get('hidden_layer_sizes', (100,))) > 2:
                mlp_params['hidden_layer_sizes'] = random.choice(simple_layers)
            if mlp_params.get('max_iter', 200) > 500:
                mlp_params['max_iter'] = random.randint(200, 500)
        
        return MLPRegressor(
            hidden_layer_sizes=mlp_params.get('hidden_layer_sizes', (100,)),
            activation=mlp_params.get('activation', 'relu'),
            solver=mlp_params.get('solver', 'adam'),
            alpha=mlp_params.get('alpha', 0.0001),
            learning_rate=mlp_params.get('learning_rate', 'constant'),
            learning_rate_init=mlp_params.get('learning_rate_init', 0.001),
            max_iter=mlp_params.get('max_iter', 200),
            beta_1=mlp_params.get('beta_1', 0.9),
            beta_2=mlp_params.get('beta_2', 0.999),
            epsilon=mlp_params.get('epsilon', 1e-8),
            random_state=RANDOM_SEED
        )
    
    def evaluate_mlp(self, individual: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, 
                    y_train: np.ndarray, y_test: np.ndarray) -> float:
        """Đánh giá fitness của MLP với các tham số."""
        # Tạo mô hình MLP Regressor sử dụng helper method
        model = self._create_mlp_model(individual)

        # Sử dụng hàm đánh giá chung cho regression
        return evaluate_regression_model(model, X_train, X_test, y_train, y_test, clip_predictions=True)
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """Chạy quá trình tối ưu hóa."""
        return self.optimizer.optimize(verbose=True)
    
    def get_best_model(self) -> MLPRegressor:
        """Lấy mô hình tốt nhất sau khi tối ưu hóa."""
        if not hasattr(self.optimizer, 'best_params') or not self.optimizer.best_params:
            raise ValueError("Chưa chạy tối ưu hóa! Hãy gọi optimize() trước.")
        
        return self._create_mlp_model(self.optimizer.best_params)

def main() -> None:
    """Hàm chính để chạy quá trình tối ưu hóa MLP."""
    try:
        # Chuẩn bị dữ liệu
        print("Bắt đầu chuẩn bị dữ liệu...")
        X, y, feature_columns = prepare_flood_data()
        
        # Lấy thông tin về đặc trưng
        feature_names, label_column = get_feature_info()
        print(f"Số lượng đặc trưng: {len(feature_names)}")
        
        # Khởi tạo và chạy bộ tối ưu RSO cho MLP
        print("Bắt đầu tối ưu hóa RSO cho MLP...")
        mlp_optimizer = MLPRSOOptimizer(X, y)
        
        start_time = time.time()
        best_params, best_score = mlp_optimizer.optimize()
        end_time = time.time()
        
        print(f"Thời gian tối ưu hóa: {end_time - start_time:.2f} giây")
        
        # In kết quả cuối cùng
        print("\n=== Kết quả cuối cùng ===")
        print(f"Điểm số tốt nhất: {best_score:.4f}")
        print("\nTham số tối ưu:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
