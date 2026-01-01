import numpy as np
import random
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from config.model_params import get_param_ranges
from evaluation.evaluation_utils import evaluate_regression_model
from optimization.model_utils import generate_random_params, create_model

class RandomizedSearch:
    def __init__(self, X, y, model_name, random_state=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.model_name = model_name.lower()
        self.random_state = random_state
        self.param_ranges = get_param_ranges(self.model_name)
        self.best_params = None
        self.best_score = -np.inf
        self.iteration_results = []
        self.iteration_counter = 0
        random.seed(random_state)
        np.random.seed(random_state)
    
    def sample_params(self):
        """Sample tham số ngẫu nhiên với seed khác nhau mỗi lần"""
        self.iteration_counter += 1
        # Thay đổi seed mỗi lần sample
        np.random.seed(self.random_state + self.iteration_counter)
        random.seed(self.random_state + self.iteration_counter)
        return generate_random_params(self.param_ranges)
    
    def _create_model(self, params):
        return create_model(self.model_name, params, self.random_state)
    
    def _evaluate_params(self, params):
        model = self._create_model(params)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=None)
        result = evaluate_regression_model(model, X_train, X_test, y_train, y_test, return_detailed=True)
        return result
    
    def save_results_to_csv(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"randomized_search_results_{timestamp}.csv"
        df = pd.DataFrame(self.iteration_results)
        df.to_csv(filename, index=False)
        return filename
    
    def search(self, n_iter=100, verbose=False):
        if verbose:
            print(f"{'Vong':>5} {'Fitness':>12} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
            print("-" * 52)
        
        for i in range(n_iter):
            params = self.sample_params()
            result = self._evaluate_params(params)
            fitness = result['fitness_score']
            iteration_result = {'iteration': i + 1, 'fitness': fitness, 'r2': result['r2'], 'mae': result['mae'], 'rmse': result['rmse']}
            for param_name, value in params.items():
                iteration_result[f'param_{param_name}'] = value
            self.iteration_results.append(iteration_result)
            if fitness > self.best_score:
                self.best_score = fitness
                self.best_params = params.copy()
            
            if verbose:
                print(f"{i+1:5d} {fitness:12.6f} {result['r2']:10.6f} {result['mae']:10.6f} {result['rmse']:10.6f}")
        
        return {'best_params': self.best_params, 'best_score': self.best_score}