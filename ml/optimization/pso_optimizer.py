import numpy as np
import random
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

from config.model_params import get_param_ranges
from evaluation.evaluation_utils import evaluate_regression_model
from optimization.model_utils import generate_random_params, create_model

class PSOOptimizer:
    def __init__(self, X, y, n_particles=10, n_iterations=100, random_state=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.1
        self.param_ranges = {}
        self.particles = []
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.best_model = None
        self.iteration_results = []
        random.seed(random_state)
        np.random.seed(random_state)
    
    def set_param_ranges(self, param_ranges):
        self.param_ranges = param_ranges
    
    def _generate_random_params(self):
        return generate_random_params(self.param_ranges)
    
    def _initialize_swarm(self):
        self.particles = []        
        for _ in range(self.n_particles):
            position = self._generate_random_params()
            particle = {'position': position, 'velocity': {}, 'best_position': position.copy(), 'best_score': -np.inf}
            for param in self.param_ranges:
                if self.param_ranges[param]['type'] in ['int', 'float', 'log_uniform']:
                    particle['velocity'][param] = 0.0
                else:
                    particle['velocity'][param] = None
            self.particles.append(particle)
    
    def _evaluate_particle(self, params):
        model = self._create_model(params)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        result = evaluate_regression_model(model, X_train, X_test, y_train, y_test, return_detailed=True)
        return result['fitness_score']
    
    def _create_model(self, params):
        raise NotImplementedError("Subclass phải implement _create_model method")
    
    def _get_detailed_metrics(self, params):
        model = self._create_model(params)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        metrics = evaluate_regression_model(model, X_train, X_test, y_train, y_test, return_detailed=True)
        return metrics
    
    def _update_particles(self):
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
                    r1, r2 = np.random.random(), np.random.random()
                    cognitive = self.c1 * r1 * (particle['best_position'][param] - particle['position'][param])
                    social = self.c2 * r2 * (self.global_best_position[param] - particle['position'][param])
                    particle['velocity'][param] = self.w * particle['velocity'][param] + cognitive + social
                    particle['position'][param] += particle['velocity'][param]
                    param_range = self.param_ranges[param]
                    particle['position'][param] = np.clip(particle['position'][param], param_range['min'], param_range['max'])
                    if param_type == 'int':
                        particle['position'][param] = int(round(particle['position'][param]))
                else:
                    if np.random.random() < 0.1:
                        particle['position'][param] = self._generate_random_params()[param]
    
    def save_results_to_csv(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pso_optimization_results_{timestamp}.csv"
        df = pd.DataFrame(self.iteration_results)
        df.to_csv(filename, index=False)
        return filename
    
    def save_best_model(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pso_best_model_{timestamp}.joblib"
        joblib.dump(self.best_model, filename)
        return filename
    
    def optimize(self, verbose=False):
        if verbose:
            print(f"{'Vong':>5} {'Fitness':>12} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
            print("-" * 52)
        
        self._initialize_swarm()
        for iteration in range(self.n_iterations):
            for particle in self.particles:
                score = self._evaluate_particle(particle['position'])
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle['position'].copy()
                    self.best_model = self._create_model(self.global_best_position)
                    self.best_model.fit(self.X, self.y)
            metrics = self._get_detailed_metrics(self.global_best_position)
            iteration_result = {'iteration': iteration + 1, 'fitness': metrics['fitness'], 'r2': metrics['r2'], 'mae': metrics['mae'], 'rmse': metrics['rmse']}
            for param_name, value in self.global_best_position.items():
                iteration_result[f'best_{param_name}'] = value
            self.iteration_results.append(iteration_result)
            
            if verbose:
                print(f"{iteration+1:5d} {metrics['fitness']:12.6f} {metrics['r2']:10.6f} {metrics['mae']:10.6f} {metrics['rmse']:10.6f}")
            
            self._update_particles()
            self.w = max(self.w_min, self.w - (self.w - self.w_min) / self.n_iterations)
        return {'best_params': self.global_best_position, 'best_score': self.global_best_score, 'best_model': self.best_model}

class RandomForestPSOOptimizer(PSOOptimizer):
    def __init__(self, X, y, n_particles=20, n_iterations=50, random_state=42):
        super().__init__(X, y, n_particles, n_iterations, random_state)
        self.set_param_ranges(get_param_ranges('rf'))
    
    def _create_model(self, params):
        return create_model('rf', params, self.random_state)

class XGBoostPSOOptimizer(PSOOptimizer):
    def __init__(self, X, y, n_particles=20, n_iterations=50, random_state=42):
        super().__init__(X, y, n_particles, n_iterations, random_state)
        self.set_param_ranges(get_param_ranges('xgb'))
    
    def _create_model(self, params):
        return create_model('xgb', params, self.random_state)

class SVMPSOOptimizer(PSOOptimizer):
    def __init__(self, X, y, n_particles=20, n_iterations=50, random_state=42):
        super().__init__(X, y, n_particles, n_iterations, random_state)
        self.set_param_ranges(get_param_ranges('svm'))
    
    def _create_model(self, params):
        return create_model('svm', params, self.random_state)

def create_optimizer(model_name, X, y, n_particles=20, n_iterations=50, random_state=42):
    model_name = model_name.lower()
    if model_name in ['rf', 'random_forest']:
        return RandomForestPSOOptimizer(X, y, n_particles, n_iterations, random_state)
    elif model_name in ['xgb', 'xgboost']:
        return XGBoostPSOOptimizer(X, y, n_particles, n_iterations, random_state)
    elif model_name in ['svm', 'support_vector_machine']:
        return SVMPSOOptimizer(X, y, n_particles, n_iterations, random_state)
    else:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ. Chỉ hỗ trợ: rf, xgb, svm")

def quick_optimize(model_name, X, y, n_particles=20, n_iterations=50, random_state=42):
    optimizer = create_optimizer(model_name, X, y, n_particles, n_iterations, random_state)
    return optimizer.optimize()