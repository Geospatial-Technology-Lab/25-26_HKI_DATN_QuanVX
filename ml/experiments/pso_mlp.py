import numpy as np
from sklearn.neural_network import MLPRegressor
import warnings
from pso_optimizer import PSOOptimizer
from model_params import get_param_ranges, OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model
from data_preprocessing import prepare_flood_data, get_feature_info
warnings.filterwarnings('ignore')

class MLPPSOOptimizer:
    def __init__(self, X, y, n_particles=None, n_iterations=None):
        if n_particles is None:
            n_particles = OPTIMIZATION_CONFIG.get('n_particles', 10)
        if n_iterations is None:
            n_iterations = OPTIMIZATION_CONFIG.get('n_iterations', 100)
            
        self.optimizer = PSOOptimizer(X, y, model_type='regression', 
                                     n_particles=n_particles, 
                                     n_iterations=n_iterations)
        
        param_ranges = get_param_ranges('mlp')
        self.optimizer.set_param_ranges(param_ranges)
        self.optimizer.set_evaluate_function(self.evaluate_mlp)
        
        # Thêm method để PSOOptimizer có thể lấy detailed metrics
        self.optimizer._create_model_for_evaluation = self._create_mlp_model
    

    def _create_mlp_model(self, params):
        mlp_params = {
            'hidden_layer_sizes': params['hidden_layer_sizes'],
            'activation': params['activation'],
            'solver': params['solver'],
            'alpha': params['alpha'],
            'learning_rate': params.get('learning_rate', 'constant'),
            'learning_rate_init': params.get('learning_rate_init', 0.001),
            'max_iter': params.get('max_iter', 500),
            'tol': params.get('tol', 1e-4),
            'random_state': 42
        }
        
        if params['solver'] == 'adam':
            mlp_params['beta_1'] = params.get('beta_1', 0.9)
            mlp_params['beta_2'] = params.get('beta_2', 0.999)
            mlp_params['epsilon'] = params.get('epsilon', 1e-8)
        
        if 'validation_fraction' in params:
            mlp_params['validation_fraction'] = params['validation_fraction']
        if 'n_iter_no_change' in params:
            mlp_params['n_iter_no_change'] = params['n_iter_no_change']
        
        return MLPRegressor(**mlp_params)
    
    def evaluate_mlp(self, params, X_train, X_test, y_train, y_test):
        """Hàm đánh giá cho MLP"""
        model = self._create_mlp_model(params)
        result = evaluate_regression_model(model, X_train, X_test, y_train, y_test, 
                                         clip_predictions=True, return_detailed=True)
        # PSOOptimizer cần một score số, không phải dict
        return result['fitness_score']
    
    def optimize(self):
        """Chạy tối ưu hóa PSO"""
        return self.optimizer.optimize()
    
    def get_best_model(self):
        """Lấy model tốt nhất"""
        best_params = self.optimizer.get_best_params()
        return self._create_mlp_model(best_params)

def main():
    try:
        print("Chuẩn bị dữ liệu...")
        X, y, _ = prepare_flood_data()
        
        feature_names, _ = get_feature_info()
        print(f"Số đặc trưng: {len(feature_names)}")
        
        print("Bắt đầu tối ưu hóa MLP...")
        mlp_optimizer = MLPPSOOptimizer(X, y)
        best_params, best_score = mlp_optimizer.optimize()
        
        print(f"\nĐiểm tốt nhất: {-best_score:.4f}")
        print("Tham số tối ưu:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        best_model = mlp_optimizer.get_best_model()
        print("\nMô hình tối ưu MLP đã được tạo thành công!")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()