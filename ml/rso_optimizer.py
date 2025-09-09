import numpy as np
import random
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Import từ các module có sẵn
from model_params import get_param_ranges
from evaluation_utils import evaluate_regression_model
from config import FLOOD_DATA_CONFIG

class RandomizedSearch:
    def __init__(self, X, y, model_name, model_type='regression', random_state=42):
        """
        Khởi tạo Randomized Search
        
        Args:
            X, y: Dữ liệu training
            model_name: Tên model ('rf', 'xgb', 'svm', 'mlp')
            model_type: Loại model ('regression' hoặc 'classification')
            random_state (int): Seed cho random
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.model_name = model_name.lower()
        self.model_type = model_type
        self.random_state = random_state
        
        # Lấy param ranges từ model_params.py
        self.param_ranges = get_param_ranges(self.model_name)
        
        # Khởi tạo các biến lưu trữ kết quả
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None
        self.history = []
        self.iteration_results = []
        
        # Thiết lập random seed
        random.seed(random_state)
        np.random.seed(random_state)
        
        print(f"Kích thước dữ liệu: {self.X.shape}")
        print(f"Phân bố nhãn: {np.bincount(self.y.astype(int))}")
        print(f"Tỷ lệ class 1: {np.mean(self.y):.3f}")
    
    def sample_params(self):
        """
        Tạo ngẫu nhiên một bộ tham số từ không gian tìm kiếm
        
        Returns:
            dict: Bộ tham số ngẫu nhiên
        """
        params = {}
        for param_name, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param_name] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                params[param_name] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                params[param_name] = 10 ** np.random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                params[param_name] = random.choice(range_info['options'])
        return params
    
    def _create_model(self, params):
        """Tạo model từ tham số"""
        try:
            if self.model_name in ['rf', 'random_forest']:
                return RandomForestRegressor(**params, random_state=self.random_state, n_jobs=-1)
            elif self.model_name in ['xgb', 'xgboost']:
                xgb_params = params.copy()
                xgb_params.update({
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbosity': 0
                })
                return xgb.XGBRegressor(**xgb_params)
            elif self.model_name in ['svm', 'support_vector_machine']:
                svm_params = self._clean_svm_params(params.copy())
                # SVR KHÔNG hỗ trợ random_state - bỏ dòng này
                # svm_params['random_state'] = self.random_state  # ← BỎ DÒNG NÀY
                return SVR(**svm_params)
            elif self.model_name in ['mlp', 'neural_network', 'multi_layer_perceptron']:
                mlp_params = self._clean_mlp_params(params.copy())
                mlp_params.update({
                    'random_state': self.random_state,
                    'early_stopping': True
                })
                return MLPRegressor(**mlp_params)
            else:
                raise ValueError(f"Model '{self.model_name}' không được hỗ trợ")
        except Exception as e:
            print(f"Lỗi khi tạo model với params {params}: {e}")
            raise
    
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
    
    def _clean_mlp_params(self, params):
        """Điều chỉnh tham số MLP dựa trên solver"""
        solver = params.get('solver', 'adam')
        
        if solver == 'lbfgs':
            for param in ['learning_rate_init', 'learning_rate', 'beta_1', 'beta_2', 'epsilon']:
                params.pop(param, None)
        elif solver == 'sgd':
            for param in ['beta_1', 'beta_2', 'epsilon']:
                params.pop(param, None)
        
        return params
    
    def _evaluate_params(self, params):
        """Đánh giá một bộ tham số"""
        try:
            # Tạo model
            model = self._create_model(params)
            
            # Chia dữ liệu train/test
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.random_state
            )
            
            # Đánh giá model
            result = evaluate_regression_model(
                model, X_train, X_test, y_train, y_test, 
                return_detailed=True
            )
            
            return result
            
        except Exception as e:
            print(f"Lỗi khi đánh giá tham số {params}: {e}")
            return {
                'fitness': -np.inf,
                'r2': 0,
                'mae': np.inf,
                'rmse': np.inf,
                'fitness_score': -np.inf
            }
    
    def print_iteration_table(self, iteration, params, fitness, r2, mae, rmse):
        """
        In kết quả của một vòng lặp theo định dạng bảng
        """
        if iteration == 1:
            print(f"{'Vòng':<6} {'Fitness':<12} {'R²':<12} {'MAE':<12} {'RMSE':<12}")
            print("-" * 60)
        
        print(f"{iteration:<6} {fitness:<12.6f} {r2:<12.6f} {mae:<12.6f} {rmse:<12.6f}")
    
    def save_results_to_csv(self, filename=None):
        """Lưu kết quả từng vòng lặp ra file CSV"""
        if not self.iteration_results:
            print("Không có dữ liệu để lưu!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"randomized_search_results_{timestamp}.csv"
        
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
            filename = f"puma_best_model_{timestamp}.joblib"
        
        # Sử dụng joblib thay vì pickle
        joblib.dump(self.best_model, filename)
        print(f"Mô hình tốt nhất đã được lưu vào: {filename}")
        
        return filename
    
    def search(self, n_iter=100, verbose=True, print_table=True, save_csv=True, save_model=True):
        """
        Thực hiện tìm kiếm ngẫu nhiên
        
        Args:
            n_iter (int): Số vòng lặp
            verbose (bool): In thông tin chi tiết
            print_table (bool): In bảng kết quả từng vòng lặp
            save_csv (bool): Lưu kết quả ra CSV
            save_model (bool): Lưu mô hình tốt nhất
        
        Returns:
            dict: Chứa best_params, best_score, best_model, csv_file, model_file
        """
        if verbose:
            print(f"\nBắt đầu Randomized Search cho {self.model_name.upper()}...")
            if print_table:
                print(f"{'Vòng':<6} {'Fitness':<12} {'R²':<12} {'MAE':<12} {'RMSE':<12}")
                print("-" * 60)
        
        for i in range(n_iter):
            # Tạo tham số ngẫu nhiên
            params = self.sample_params()
            
            # Đánh giá tham số
            result = self._evaluate_params(params)
            
            fitness = result['fitness_score']
            r2 = result['r2']
            mae = result['mae']
            rmse = result['rmse']
            
            # Lưu kết quả vòng lặp
            iteration_result = {
                'iteration': i + 1,
                'fitness': fitness,
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
            
            # Thêm tham số vào kết quả
            for param_name, value in params.items():
                iteration_result[f'param_{param_name}'] = value
            
            self.iteration_results.append(iteration_result)
            
            # In bảng kết quả nếu được yêu cầu
            if verbose and print_table:
                self.print_iteration_table(i + 1, params, fitness, r2, mae, rmse)
            
            # Cập nhật best score và mô hình tốt nhất
            if fitness > self.best_score:
                self.best_score = fitness
                self.best_params = params.copy()
                
                # Cập nhật mô hình tốt nhất
                try:
                    self.best_model = self._create_model(params)
                    self.best_model.fit(self.X, self.y)
                except Exception as e:
                    print(f"Lỗi khi tạo mô hình tốt nhất: {e}")
                
                if verbose and not print_table:
                    print(f"Vòng {i+1}: Tìm được tham số tốt hơn - Fitness: {fitness:.6f}")
        
        if verbose:
            print("\nRandomized Search hoàn thành!")
            print("=" * 60)
            print("KẾT QUẢ TỐT NHẤT")
            print("=" * 60)
            print(f"Score: {self.best_score:.6f}")
            print("\nTham số tốt nhất:")
            # KIỂM TRA None trước khi gọi .items()
            if self.best_params is not None:
                for param_name, value in self.best_params.items():
                    print(f"{param_name}: {value}")
            else:
                print("Không tìm được tham số tốt nhất (tất cả iterations đều lỗi)")
            print("=" * 60)
        
        # Lưu file nếu được yêu cầu
        csv_filename = None
        model_filename = None
        
        if save_csv:
            csv_filename = self.save_results_to_csv()
        
        if save_model and self.best_model is not None:
            model_filename = self.save_best_model()
        
        return {
            'best_params': self.best_params, 
            'best_score': self.best_score,
            'best_model': self.best_model,
            'csv_file': csv_filename,
            'model_file': model_filename
        }
    
    def reset(self):
        """Reset trạng thái tìm kiếm"""
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None
        self.history = []
        self.iteration_results = []
    
    def get_best_params(self):
        """Trả về tham số tốt nhất"""
        return self.best_params
    
    def get_best_score(self):
        """Trả về điểm số tốt nhất"""
        return self.best_score
    
    def get_best_model(self):
        """Trả về mô hình tốt nhất"""
        return self.best_model
    
    def get_results_dataframe(self):
        """
        Trả về DataFrame chứa kết quả tất cả vòng lặp
        
        Returns:
            pd.DataFrame: Kết quả tất cả vòng lặp
        """
        if not self.iteration_results:
            return None
        return pd.DataFrame(self.iteration_results)


# Hàm tiện ích để tạo optimizer cho model cụ thể
def create_randomized_search_optimizer(model_name, X, y, model_type='regression', random_state=42):
    """
    Tạo Randomized Search optimizer cho model cụ thể
    
    Args:
        model_name: Tên model ('rf', 'xgb', 'svm', 'mlp')
        X, y: Dữ liệu training
        model_type: 'classification' hoặc 'regression'
        random_state: Seed cho random
    
    Returns:
        RandomizedSearch instance
    """
    return RandomizedSearch(X, y, model_name, model_type, random_state)


# Hàm tiện ích để chạy tối ưu hóa nhanh
def quick_randomized_search(model_name, X, y, model_type='regression', n_iter=100, 
                           verbose=True, save_files=True, random_state=42):
    """
    Chạy Randomized Search nhanh cho một model
    
    Args:
        model_name: Tên model
        X, y: Dữ liệu
        model_type: Loại model
        n_iter: Số vòng lặp
        verbose: In thông tin chi tiết
        save_files: Lưu file kết quả
        random_state: Seed cho random
    
    Returns:
        dict: Chứa best_params, best_score, best_model, csv_file, model_file
    """
    # Tạo optimizer
    optimizer = create_randomized_search_optimizer(model_name, X, y, model_type, random_state)
    
    # Chạy tối ưu hóa
    result = optimizer.search(n_iter=n_iter, verbose=verbose, save_csv=save_files, save_model=save_files)
    
    return result


def create_param_range(param_type, **kwargs):
    """
    Tạo định nghĩa phạm vi tham số
    
    Args:
        param_type (str): Loại tham số
        **kwargs: Các tham số bổ sung
    
    Returns:
        dict: Định nghĩa phạm vi tham số
    """
    range_def = {'type': param_type}
    range_def.update(kwargs)
    return range_def


# Export các class và function chính
__all__ = [
    'RandomizedSearch',
    'create_randomized_search_optimizer',
    'quick_randomized_search',
    'create_param_range'
]