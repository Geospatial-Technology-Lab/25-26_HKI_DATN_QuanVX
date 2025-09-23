"""PSO Optimizer - Bộ tối ưu hóa Particle Swarm Optimization chung."""

import numpy as np
import random
import warnings
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb

# Import từ các module có sẵn
from model_params import get_param_ranges, OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model

warnings.filterwarnings('ignore')

# Thiết lập hạt giống cố định
RANDOM_SEED = OPTIMIZATION_CONFIG['random_seed']
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Cấu hình PSO từ OPTIMIZATION_CONFIG
PSO_CONFIG = {
    'w': 0.9,        # Trọng số quán tính
    'c1': 2.0,       # Tham số nhận thức
    'c2': 2.0,       # Tham số xã hội
    'w_min': 0.1     # Trọng số quán tính tối thiểu
}

class PSOOptimizer:
    """Bộ tối ưu hóa PSO chung cho tất cả các mô hình Machine Learning."""
    
    def __init__(self, X, y, model_type='regression', n_particles=None, n_iterations=None):
        """Khởi tạo bộ tối ưu hóa PSO."""
        self.X = np.array(X)
        self.y = np.array(y)
        self.model_type = model_type
        
        # Sử dụng cấu hình mặc định nếu không được cung cấp
        self.n_particles = n_particles or OPTIMIZATION_CONFIG['population_size']
        self.n_iterations = n_iterations or OPTIMIZATION_CONFIG['generations']
        
        # Tham số PSO
        self.w = PSO_CONFIG['w']
        self.c1 = PSO_CONFIG['c1']
        self.c2 = PSO_CONFIG['c2']
        self.w_min = PSO_CONFIG['w_min']
        
        # Khởi tạo các biến
        self.param_ranges = {}
        self.particles = []
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.optimization_history = []
        self.avg_scores_history = []
        
        # Biến lưu mô hình tốt nhất
        self.best_model = None
        self.iteration_results = []  # Lưu kết quả từng vòng lặp
        
        print(f"Kích thước dữ liệu: {self.X.shape}")
        print(f"Phân bố nhãn: {np.bincount(self.y.astype(int))}")
        print(f"Tỷ lệ class 1: {np.mean(self.y):.3f}")
    
    def set_param_ranges(self, param_ranges):
        """Thiết lập phạm vi tham số"""
        self.param_ranges = param_ranges
    
    def _generate_random_params(self):
        """Tạo bộ tham số ngẫu nhiên"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
            elif range_info['type'] == 'float':
                params[param] = np.random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                params[param] = 10 ** np.random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                params[param] = random.choice(range_info['options'])
        
        return params
    
    def _initialize_swarm(self):
        """Khởi tạo bầy đàn hạt"""
        self.particles = []        
        for _ in range(self.n_particles):
            position = self._generate_random_params()
            particle = {
                'position': position,
                'velocity': {},
                'best_position': position.copy(),
                'best_score': -np.inf
            }
            
            # Khởi tạo vận tốc
            for param in self.param_ranges:
                if self.param_ranges[param]['type'] in ['int', 'float', 'log_uniform']:
                    particle['velocity'][param] = 0.0
                else:
                    particle['velocity'][param] = None
            
            self.particles.append(particle)
    
    def _evaluate_particle(self, params):
        """Đánh giá một hạt với bộ tham số cho trước"""
        try:
            # Tạo model dựa trên params
            model = self._create_model(params)
            
            # Sử dụng cross validation
            if self.model_type == 'classification':
                scoring = 'f1' if len(np.unique(self.y)) == 2 else 'f1_macro'
                scores = cross_val_score(model, self.X, self.y, cv=5, scoring=scoring, n_jobs=1)
                return float(np.mean(scores))
            else:
                # Cho regression, sử dụng hàm evaluate có sẵn
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
                )
                result = evaluate_regression_model(model, X_train, X_test, y_train, y_test, 
                                                 return_detailed=True)
                return result['fitness_score']
                
        except Exception as e:
            print(f"Lỗi trong đánh giá: {str(e)}")
            return -np.inf
    
    def _create_model(self, params):
        """Tạo model từ tham số - cần được override bởi subclass"""
        raise NotImplementedError("Subclass phải implement _create_model method")
    
    def _get_detailed_metrics(self, params):
        """Lấy các metrics chi tiết cho regression"""
        try:
            model = self._create_model(params)
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
            )
            metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, return_detailed=True)
            return metrics
        except Exception as e:
            return {'fitness': -np.inf, 'r2': 0, 'mae': np.inf, 'rmse': np.inf, 'fitness_score': -np.inf}
    
    def _update_particles(self):
        """Cập nhật vị trí và vận tốc của các hạt"""
        for particle in self.particles:
            
            if not self.global_best_position:
                continue

            for param in self.param_ranges:
                param_type = self.param_ranges[param]['type']
                
                if param_type in ['int', 'float', 'log_uniform']:

                    if param not in particle['best_position']:
                        particle['best_position'][param] = particle['position'][param]
                
                    if param not in self.global_best_position:
                        self.global_best_position[param] = particle['position'][param]

                    # Cập nhật vận tốc cho tham số số
                    r1, r2 = np.random.random(), np.random.random()
                    
                    cognitive = self.c1 * r1 * (particle['best_position'][param] - particle['position'][param])
                    social = self.c2 * r2 * (self.global_best_position[param] - particle['position'][param])
                    
                    particle['velocity'][param] = (
                        self.w * particle['velocity'][param] + cognitive + social
                    )
                    
                    # Cập nhật vị trí
                    particle['position'][param] += particle['velocity'][param]
                    
                    # Ràng buộc trong phạm vi
                    param_range = self.param_ranges[param]
                    particle['position'][param] = np.clip(
                        particle['position'][param], 
                        param_range['min'], 
                        param_range['max']
                    )
                    
                    # Làm tròn cho tham số integer
                    if param_type == 'int':
                        particle['position'][param] = int(round(particle['position'][param]))
                        
                else:
                    # Đối với tham số categorical, thỉnh thoảng chọn ngẫu nhiên
                    if np.random.random() < 0.1:  # 10% cơ hội
                        particle['position'][param] = self._generate_random_params()[param]
    
    def save_results_to_csv(self, filename=None):
        """Lưu kết quả từng vòng lặp ra file CSV"""
        if not self.iteration_results:
            print("Không có dữ liệu để lưu!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pso_optimization_results_{timestamp}.csv"
        
        df = pd.DataFrame(self.iteration_results)
        df.to_csv(filename, index=False)
        print(f"Kết quả đã được lưu vào: {filename}")
        
        return filename
    
    def save_best_model(self, filename=None):
        """Lưu mô hình tốt nhất"""
        if self.best_model is None:
            print("Chưa có mô hình tốt nhất để lưu!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pso_best_model_{timestamp}.joblib"  # Đổi extension
        
        joblib.dump(self.best_model, filename)
        print(f"Mô hình tốt nhất đã được lưu vào: {filename}")
        
        return filename
    
    def optimize(self, verbose=True, print_table=True, save_csv=True, save_model=True):
        """Chạy thuật toán PSO"""
        if not self.param_ranges:
            raise ValueError("Chưa thiết lập phạm vi tham số! Hãy gọi set_param_ranges() trước.")
        
        # Khởi tạo bầy đàn
        self._initialize_swarm()
        
        if verbose:
            print("\nBắt đầu tối ưu hóa PSO...")
            if print_table:
                print(f"{'Vòng':<6} {'Fitness':<12} {'R2':<12} {'MAE':<12} {'RMSE':<12}")
                print("-" * 60)
        
        for iteration in range(self.n_iterations):
            avg_score = 0
            
            # Đánh giá từng hạt
            for particle in self.particles:
                score = self._evaluate_particle(particle['position'])
                avg_score += score
                
                # Cập nhật best cục bộ
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # Cập nhật best toàn cục và mô hình tốt nhất
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle['position'].copy()
                    
                    # Cập nhật mô hình tốt nhất
                    try:
                        self.best_model = self._create_model(self.global_best_position)
                        self.best_model.fit(self.X, self.y)
                    except Exception as e:
                        print(f"Lỗi khi tạo mô hình tốt nhất: {e}")
            
            # Tính điểm và metrics chi tiết cho best position hiện tại
            if self.model_type == 'regression':
                metrics = self._get_detailed_metrics(self.global_best_position)
                
                # Lưu kết quả vòng lặp
                iteration_result = {
                    'iteration': iteration + 1,
                    'fitness': metrics['fitness'],
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'avg_fitness': avg_score / self.n_particles if self.n_particles > 0 else -np.inf
                }
                
                # Thêm tham số tốt nhất vào kết quả
                for param_name, value in self.global_best_position.items():
                    iteration_result[f'best_{param_name}'] = value
                
                self.iteration_results.append(iteration_result)
                
                # Lưu lịch sử
                self.optimization_history.append(metrics['fitness'])
                self.avg_scores_history.append(avg_score / self.n_particles if self.n_particles > 0 else -np.inf)
                
                if verbose and print_table:
                    print(f"{iteration+1:<6} {metrics['fitness']:<12.6f} {metrics['r2']:<12.6f} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f}")
            else:
                # Đối với classification, giữ nguyên output cũ
                avg_score = avg_score / self.n_particles if self.n_particles > 0 else -np.inf
                
                iteration_result = {
                    'iteration': iteration + 1,
                    'fitness': self.global_best_score,
                    'avg_fitness': avg_score
                }
                
                for param_name, value in self.global_best_position.items():
                    iteration_result[f'best_{param_name}'] = value
                
                self.iteration_results.append(iteration_result)
                self.optimization_history.append(self.global_best_score)
                self.avg_scores_history.append(avg_score)
                
                if verbose and print_table:
                    print(f"{iteration+1:<6} {self.global_best_score:<12.6f} {avg_score:<12.6f}")
            
            # Cập nhật vị trí và vận tốc
            self._update_particles()
            
            # Giảm trọng số quán tính theo thời gian
            self.w = max(self.w_min, self.w - (self.w - self.w_min) / self.n_iterations)
        
        if verbose:
            print("\nTối ưu hóa PSO hoàn thành!")
            print("=" * 60)
            print("KẾT QUẢ TỐT NHẤT")
            print("=" * 60)
            print(f"Score: {self.global_best_score:.6f}")
            print("\nTham số tốt nhất:")
            for param_name, value in self.global_best_position.items():
                print(f"{param_name}: {value}")
            print("=" * 60)
        
        # Lưu file nếu được yêu cầu
        csv_filename = None
        model_filename = None
        
        if save_csv:
            csv_filename = self.save_results_to_csv()
        
        if save_model:
            model_filename = self.save_best_model()
        
        return {
            'best_params': self.global_best_position, 
            'best_score': self.global_best_score,
            'best_model': self.best_model,
            'csv_file': csv_filename,
            'model_file': model_filename
        }
    
    def get_best_params(self):
        """Lấy tham số tốt nhất"""
        return self.global_best_position
    
    def get_best_score(self):
        """Lấy điểm số tốt nhất"""
        return self.global_best_score
    
    def get_best_model(self):
        """Lấy mô hình tốt nhất"""
        return self.best_model


class RandomForestPSOOptimizer(PSOOptimizer):
    """PSO Optimizer cho Random Forest"""
    
    def __init__(self, X, y, model_type='regression', n_particles=None, n_iterations=None):
        super().__init__(X, y, model_type, n_particles, n_iterations)
        self.set_param_ranges(get_param_ranges('rf'))
    
    def _create_model(self, params):
        """Tạo Random Forest model"""
        if self.model_type == 'classification':
            return RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1)
        else:
            return RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)


class XGBoostPSOOptimizer(PSOOptimizer):
    """PSO Optimizer cho XGBoost"""
    
    def __init__(self, X, y, model_type='regression', n_particles=None, n_iterations=None):
        super().__init__(X, y, model_type, n_particles, n_iterations)
        self.set_param_ranges(get_param_ranges('xgb'))
    
    def _create_model(self, params):
        """Tạo XGBoost model"""
        xgb_params = params.copy()
        xgb_params.update({
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        })
        
        if self.model_type == 'classification':
            return xgb.XGBClassifier(**xgb_params)
        else:
            return xgb.XGBRegressor(**xgb_params)


class SVMPSOOptimizer(PSOOptimizer):
    """PSO Optimizer cho SVM"""
    
    def __init__(self, X, y, model_type='regression', n_particles=None, n_iterations=None):
        super().__init__(X, y, model_type, n_particles, n_iterations)
        self.set_param_ranges(get_param_ranges('svm'))
    
    def _create_model(self, params):
        svm_params = self._clean_svm_params(params.copy())
        
        if self.model_type == 'classification':
            svm_params['random_state'] = RANDOM_SEED
            return SVC(**svm_params)
        else:
            return SVR(**svm_params)
    
    def _clean_svm_params(self, params):
        """Điều chỉnh tham số SVM dựa trên kernel"""
        kernel = params.get('kernel', 'rbf')
        
        if kernel == 'linear':
            params.pop('gamma', None)
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'rbf':
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'sigmoid':
            params.pop('degree', None)
        
        return params


class MLPPSOOptimizer(PSOOptimizer):
    """PSO Optimizer cho MLP"""
    
    def __init__(self, X, y, model_type='regression', n_particles=None, n_iterations=None):
        super().__init__(X, y, model_type, n_particles, n_iterations)
        self.set_param_ranges(get_param_ranges('mlp'))
    
    def _create_model(self, params):
        """Tạo MLP model"""
        mlp_params = self._clean_mlp_params(params.copy())
        mlp_params.update({
            'random_state': RANDOM_SEED,
            'early_stopping': True
        })
        
        if self.model_type == 'classification':
            return MLPClassifier(**mlp_params)
        else:
            return MLPRegressor(**mlp_params)
    
    def _clean_mlp_params(self, params):
        """Điều chỉnh tham số MLP dựa trên solver"""
        solver = params.get('solver', 'adam')
        
        if solver == 'lbfgs':
            # Loại bỏ các tham số không được hỗ trợ bởi lbfgs
            for param in ['learning_rate_init', 'learning_rate', 'beta_1', 'beta_2', 'epsilon']:
                params.pop(param, None)
        elif solver == 'sgd':
            # Loại bỏ các tham số chỉ dành cho adam
            for param in ['beta_1', 'beta_2', 'epsilon']:
                params.pop(param, None)
        
        return params


# Hàm tiện ích để tạo optimizer
def create_optimizer(model_name, X, y, model_type='regression', n_particles=None, n_iterations=None):
    """
    Tạo optimizer cho model cụ thể
    
    Args:
        model_name: Tên model ('rf', 'xgb', 'svm', 'mlp')
        X, y: Dữ liệu training
        model_type: 'classification' hoặc 'regression'
        n_particles: Số lượng particles
        n_iterations: Số vòng lặp
    
    Returns:
        Optimizer instance
    """
    model_name = model_name.lower()
    
    if model_name in ['rf', 'random_forest']:
        return RandomForestPSOOptimizer(X, y, model_type, n_particles, n_iterations)
    elif model_name in ['xgb', 'xgboost']:
        return XGBoostPSOOptimizer(X, y, model_type, n_particles, n_iterations)
    elif model_name in ['svm', 'support_vector_machine']:
        return SVMPSOOptimizer(X, y, model_type, n_particles, n_iterations)
    elif model_name in ['mlp', 'neural_network', 'multi_layer_perceptron']:
        return MLPPSOOptimizer(X, y, model_type, n_particles, n_iterations)
    else:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ. "
                        f"Các model hỗ trợ: rf, xgb, svm, mlp")


# Hàm tiện ích để chạy tối ưu hóa nhanh
def quick_optimize(model_name, X, y, model_type='regression', verbose=True, save_files=True):   
    # Tạo optimizer
    optimizer = create_optimizer(model_name, X, y, model_type)
    
    # Chạy tối ưu hóa
    result = optimizer.optimize(verbose=verbose, save_csv=save_files, save_model=save_files)
    
    return result


# Export các class và function chính
__all__ = [
    'PSOOptimizer',
    'RandomForestPSOOptimizer', 
    'XGBoostPSOOptimizer',
    'SVMPSOOptimizer',
    'MLPPSOOptimizer',
    'create_optimizer',
    'quick_optimize'
]