import numpy as np
import random
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from config.model_params import get_param_ranges
from evaluation.evaluation_utils import evaluate_regression_model
from optimization.model_utils import generate_random_params, create_model

class PSOOptimizer:
    """Particle Swarm Optimization"""
    def __init__(self, X, y, n_particles=10, n_iterations=50, random_state=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state
        # Tham số PSO chuẩn
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_min = 0.4
        self.param_ranges = {}
        self.particles = []
        self.global_best_position = None
        self.global_best_score = -np.inf
        self.iteration_results = []
        random.seed(random_state)
        np.random.seed(random_state)
    
    def set_param_ranges(self, param_ranges):
        self.param_ranges = param_ranges
    
    def _initialize_swarm(self):
        """Khởi tạo bầy đàn hạt"""
        self.particles = []
        for _ in range(self.n_particles):
            position = generate_random_params(self.param_ranges)
            velocity = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] in ['int', 'float', 'log_uniform']:
                    velocity[param] = 0.0
                else:
                    velocity[param] = None
            
            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_score': -np.inf
            }
            self.particles.append(particle)
    
    def _evaluate_particle(self, params):
        """Đánh giá fitness của hạt"""
        model = self._create_model(params)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=None
        )
        result = evaluate_regression_model(
            model, X_train, X_test, y_train, y_test, return_detailed=True
        )
        return result
    
    def _create_model(self, params):
        raise NotImplementedError("Subclass phải implement _create_model method")
    
    def _update_particles(self):
        """Cập nhật vị trí và vận tốc của các hạt"""
        if not self.global_best_position:
            return
        
        for particle in self.particles:
            for param, range_info in self.param_ranges.items():
                param_type = range_info['type']
                
                if param_type in ['int', 'float', 'log_uniform']:
                    # Cập nhật theo công thức PSO chuẩn
                    r1, r2 = np.random.random(), np.random.random()
                    cognitive = self.c1 * r1 * (particle['best_position'][param] - particle['position'][param])
                    social = self.c2 * r2 * (self.global_best_position[param] - particle['position'][param])
                    
                    particle['velocity'][param] = self.w * particle['velocity'][param] + cognitive + social
                    particle['position'][param] += particle['velocity'][param]
                    
                    # Giới hạn trong range
                    particle['position'][param] = np.clip(
                        particle['position'][param], 
                        range_info['min'], 
                        range_info['max']
                    )
                    
                    if param_type == 'int':
                        particle['position'][param] = int(round(particle['position'][param]))
                else:
                    # Với categorical: random với xác suất nhỏ
                    if np.random.random() < 0.1:
                        particle['position'][param] = random.choice(range_info['options'])
    
    def save_results_to_csv(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pso_optimization_results_{timestamp}.csv"
        df = pd.DataFrame(self.iteration_results)
        df.to_csv(filename, index=False)
        return filename
    
    def optimize(self, verbose=True):
        """Thực hiện tối ưu hóa PSO"""
        if verbose:
            print(f"{'Vong':>5} {'Fitness':>12} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
            print("-" * 52)
        
        self._initialize_swarm()
        
        for iteration in range(self.n_iterations):
            # Đánh giá tất cả các hạt
            for particle in self.particles:
                metrics = self._evaluate_particle(particle['position'])
                score = metrics['fitness_score']
                
                # Cập nhật personal best
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # Cập nhật global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle['position'].copy()
            
            # Lưu kết quả của iteration
            iteration_result = {
                'iteration': iteration + 1,
                'fitness': self.global_best_score,
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse']
            }
            for param_name, value in self.global_best_position.items():
                iteration_result[f'best_{param_name}'] = value
            self.iteration_results.append(iteration_result)
            
            # In kết quả
            if verbose:
                print(f"{iteration+1:5d} {self.global_best_score:12.6f} {metrics['r2']:10.6f} {metrics['mae']:10.6f} {metrics['rmse']:10.6f}")
            
            # Cập nhật vị trí các hạt
            self._update_particles()
            
            # Giảm dần inertia weight
            self.w = max(self.w_min, self.w - (self.w - self.w_min) / self.n_iterations)
        
        return {
            'best_params': self.global_best_position,
            'best_score': self.global_best_score
        }

class RandomForestPSOOptimizer(PSOOptimizer):
    def __init__(self, X, y, n_particles=10, n_iterations=50, random_state=42):
        super().__init__(X, y, n_particles, n_iterations, random_state)
        self.set_param_ranges(get_param_ranges('rf'))
    
    def _create_model(self, params):
        return create_model('rf', params, self.random_state)

class XGBoostPSOOptimizer(PSOOptimizer):
    def __init__(self, X, y, n_particles=10, n_iterations=50, random_state=42):
        super().__init__(X, y, n_particles, n_iterations, random_state)
        self.set_param_ranges(get_param_ranges('xgb'))
    
    def _create_model(self, params):
        return create_model('xgb', params, self.random_state)

class SVMPSOOptimizer(PSOOptimizer):
    def __init__(self, X, y, n_particles=10, n_iterations=50, random_state=42):
        super().__init__(X, y, n_particles, n_iterations, random_state)
        self.set_param_ranges(get_param_ranges('svm'))
    
    def _create_model(self, params):
        return create_model('svm', params, self.random_state)

def create_optimizer(model_name, X, y, n_particles=10, n_iterations=50, random_state=42):
    """Tạo PSO optimizer cho model cụ thể"""
    model_name = model_name.lower()
    if model_name in ['rf', 'random_forest']:
        return RandomForestPSOOptimizer(X, y, n_particles, n_iterations, random_state)
    elif model_name in ['xgb', 'xgboost']:
        return XGBoostPSOOptimizer(X, y, n_particles, n_iterations, random_state)
    elif model_name in ['svm', 'support_vector_machine']:
        return SVMPSOOptimizer(X, y, n_particles, n_iterations, random_state)
    else:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ. Chỉ hỗ trợ: rf, xgb, svm")