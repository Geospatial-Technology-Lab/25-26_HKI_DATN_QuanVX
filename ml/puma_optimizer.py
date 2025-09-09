import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

from config import FLOOD_DATA_CONFIG
from data_preprocessing import prepare_flood_data
from evaluation_utils import evaluate_regression_model

# Thêm seed cố định để tái tạo kết quả
RANDOM_SEED = FLOOD_DATA_CONFIG['random_state']

class PUMAOptimizer:
    def __init__(self, X=None, y=None, model_type='regression', population_size=10, generations=100, random_state=None):
        """Bộ tối ưu hóa PUMA tổng quát cho nhiều loại mô hình."""
        self.model_type = model_type
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state or RANDOM_SEED
        self.best_individual = None
        self.best_score = -np.inf  # Luôn tối đa hóa fitness
        self.best_scores_history = []
        self.pCR = 0.5  # Tỷ lệ lai ghép ban đầu
        self.p = 0.1    # Tỷ lệ điều chỉnh pCR
        
        # Biến lưu mô hình tốt nhất
        self.best_model = None
        self.iteration_results = []  # Lưu kết quả từng vòng lặp

        # Experience Management System variables
        self.UnSelected = [1, 1]  # [Exploration_count, Exploitation_count]
        self.F3_Explore = 0
        self.F3_Exploit = 0
        self.Seq_Time_Explore = [1, 1, 1]
        self.Seq_Time_Exploit = [1, 1, 1]
        self.Seq_Cost_Explore = [1, 1, 1]  
        self.Seq_Cost_Exploit = [1, 1, 1]
        self.Score_Explore = 0
        self.Score_Exploit = 0
        self.PF = [0.5, 0.5, 0.3]  # Performance factors
        self.PF_F3 = []
        self.Mega_Explor = 0.99
        self.Mega_Exploit = 0.99
        self.Flag_Change = 1
        self.Initial_Best = None
        self.Costs_Explor = []
        self.Costs_Exploit = []
        self.SelectFlag = 0
        self.experienced_phase_started = False
        
        # Thiết lập random seed
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Chuẩn bị dữ liệu
        if X is not None and y is not None:
            # Sử dụng dữ liệu được cung cấp
            X_prepared, y_prepared = X, y
            self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            # Sử dụng hàm chuẩn bị dữ liệu từ data_preprocessing
            X_prepared, y_prepared, self.feature_columns = prepare_flood_data(
                config=None, shuffle_data=True, debug=False
            )

        # Chia dữ liệu
        stratify = y_prepared if model_type == 'classification' else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_prepared, y_prepared, 
            test_size=FLOOD_DATA_CONFIG['test_size'], 
            stratify=stratify, 
            random_state=self.random_state
        )

        # Kiểm tra và chuẩn hóa từng cột riêng biệt
        # Khởi tạo mảng scaled với dữ liệu gốc
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        # Duyệt qua từng cột để kiểm tra và chuẩn hóa nếu cần
        for col in range(self.X_train.shape[1]):
            # Tính statistics cho cột hiện tại
            col_min = np.min(self.X_train[:, col])
            col_max = np.max(self.X_train[:, col])
            col_range = col_max - col_min
            col_mean = np.mean(self.X_train[:, col])
            col_std = np.std(self.X_train[:, col])
            
            # Kiểm tra xem cột có cần chuẩn hóa không
            # Điều kiện: có range lớn hoặc mean/std bất thường
            need_scaling = (col_range > 10) or (abs(col_mean) > 10) or (col_std > 10)
            
            if need_scaling:
                # Thực hiện chuẩn hóa chỉ cho cột này
                if col_range != 0:
                    self.X_train_scaled[:, col] = (self.X_train[:, col] - col_min) / col_range
                    self.X_test_scaled[:, col] = (self.X_test[:, col] - col_min) / col_range
                
                print(f"Đã chuẩn hóa cột {self.feature_columns[col]} với range={col_range:.2f}, "
                      f"mean={col_mean:.2f}, std={col_std:.2f}")
        
        # Sẽ được thiết lập bởi model cụ thể
        self.param_ranges = {}
        self.evaluate_function = None
    
    def set_param_ranges(self, param_ranges):
        """Thiết lập phạm vi tham số cho mô hình cụ thể"""
        self.param_ranges = param_ranges
    
    def set_evaluate_function(self, evaluate_function):
        """Thiết lập hàm đánh giá cho mô hình cụ thể"""
        self.evaluate_function = evaluate_function

    def create_individual(self):
        """Tạo một cá thể ngẫu nhiên với các tham số trong phạm vi cho phép"""
        individual = {}
        
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                individual[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                individual[param] = 10 ** random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                individual[param] = random.choice(range_info['options'])
        
        return individual

    def create_model_from_params(self, individual):
        """Tạo model từ dictionary tham số"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        try:
            from xgboost import XGBRegressor
        except ImportError:
            XGBRegressor = None
        
        # Xác định loại model từ tham số
        if 'n_estimators' in individual:
            if 'learning_rate' in individual and XGBRegressor is not None:
                # XGBoost
                return XGBRegressor(**individual, random_state=self.random_state)
            else:
                # Random Forest
                return RandomForestRegressor(**individual, random_state=self.random_state)
        elif 'C' in individual and 'gamma' in individual:
            # SVM
            return SVR(**individual)
        elif 'hidden_layer_sizes' in individual:
            # MLP
            return MLPRegressor(**individual, random_state=self.random_state)
        else:
            # Default fallback
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
    
    def calculate_metrics(self, individual):
        """Tính toán các metric chi tiết cho regression"""
        if self.model_type != 'regression':
            return None, None, None
        
        try:
            # Tạo model từ individual parameters
            model = self.create_model_from_params(individual)
            
            # Sử dụng evaluate_regression_model từ evaluation_utils
            detailed_metrics = evaluate_regression_model(
                model, self.X_train_scaled, self.X_test_scaled, 
                self.y_train, self.y_test, 
                clip_predictions=True, return_detailed=True
            )
            
            return detailed_metrics['r2'], detailed_metrics['mae'], detailed_metrics['rmse']
            
        except Exception as e:
            return 0.0, float('inf'), float('inf')
    
    def get_detailed_metrics(self, individual):
        """Lấy metrics chi tiết dưới dạng dictionary"""
        try:
            model = self.create_model_from_params(individual)
            detailed_metrics = evaluate_regression_model(
                model, self.X_train_scaled, self.X_test_scaled, 
                self.y_train, self.y_test, 
                clip_predictions=True, return_detailed=True
            )
            return detailed_metrics
        except Exception as e:
            return {'fitness': -np.inf, 'r2': 0, 'mae': np.inf, 'rmse': np.inf, 'fitness_score': -np.inf}
    
    def evaluate_individual(self, individual):
        """Đánh giá fitness của một cá thể"""
        if self.evaluate_function is None:
            raise ValueError("Chưa thiết lập hàm đánh giá! Sử dụng set_evaluate_function()")
        return self.evaluate_function(individual, self.X_train_scaled, self.X_test_scaled, 
                                    self.y_train, self.y_test)
    
    def clip_individual(self, individual):
        """Đảm bảo các tham số nằm trong phạm vi cho phép"""
        for param, range_info in self.param_ranges.items():
            if range_info['type'] in ['int', 'float']:
                individual[param] = max(range_info['min'], 
                                      min(range_info['max'], individual[param]))
            elif range_info['type'] == 'log_uniform':
                individual[param] = max(range_info['min'], 
                                      min(range_info['max'], individual[param]))
            elif range_info['type'] == 'choice':
                if individual[param] not in range_info['options']:
                    individual[param] = random.choice(range_info['options'])
        return individual
    
    def exploration_phase(self, population, fitness_values):
        """Giai đoạn khám phá PUMA"""
        # Sắp xếp population theo fitness (tối đa hóa)
        sorted_indices = np.argsort(fitness_values)[::-1]  # Descending order
        population = [population[i] for i in sorted_indices]
        fitness_values = [fitness_values[i] for i in sorted_indices]
        
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Tạo list indices khác i
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            
            # Chọn 6 indices ngẫu nhiên (a,b,c,d,e,f)
            if len(available_indices) >= 6:
                selected_indices = random.sample(available_indices, 6)
                a, b, c, d, e, f = selected_indices
            else:
                # Nếu không đủ 6, lặp lại một số indices
                selected_indices = random.choices(available_indices, k=6)
                a, b, c, d, e, f = selected_indices
            
            # G coefficient
            G = 2 * random.random() - 1
            
            # Tạo individual mới
            new_individual = {}
            
            # Chọn j0 - tham số bắt buộc phải thay đổi
            param_keys = list(self.param_ranges.keys())
            j0 = random.choice(param_keys)
            
            for param, range_info in self.param_ranges.items():
                if param == j0 or random.random() <= self.pCR:
                    if random.random() < 0.5:
                        # Tạo giá trị ngẫu nhiên mới (Eq 25 - phần đầu)
                        if range_info['type'] == 'int':
                            new_individual[param] = random.randint(range_info['min'], range_info['max'])
                        elif range_info['type'] == 'float':
                            new_individual[param] = random.uniform(range_info['min'], range_info['max'])
                        elif range_info['type'] == 'log_uniform':
                            log_min = np.log10(range_info['min'])
                            log_max = np.log10(range_info['max'])
                            new_individual[param] = 10 ** random.uniform(log_min, log_max)
                        elif range_info['type'] == 'choice':
                            new_individual[param] = random.choice(range_info['options'])
                    else:
                        # Sử dụng công thức PUMA (Eq 25 - phần sau)
                        if range_info['type'] in ['int', 'float', 'log_uniform']:
                            term1 = population[a][param] + G * (population[a][param] - population[b][param])
                            term2 = G * (((population[a][param] - population[b][param]) - 
                                        (population[c][param] - population[d][param])) + 
                                    ((population[c][param] - population[d][param]) - 
                                        (population[e][param] - population[f][param])))
                            new_val = term1 + term2
                            
                            if range_info['type'] == 'int':
                                new_individual[param] = int(round(new_val))
                            else:
                                new_individual[param] = new_val
                        elif range_info['type'] == 'choice':
                            new_individual[param] = random.choice(range_info['options'])
                else:
                    # Giữ nguyên giá trị cũ
                    new_individual[param] = current[param]
            
            # Clip parameters
            new_individual = self.clip_individual(new_individual)
            
            # Đánh giá fitness
            new_fitness_val = self.evaluate_individual(new_individual)
            
            # So sánh và cập nhật (tối đa hóa)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Cập nhật pCR khi không cải thiện (Eq 30)
                p = (1 - self.pCR) / self.population_size
                self.pCR = min(self.pCR + p, 0.9)
        return new_population, new_fitness
    
    def exploitation_phase(self, population, fitness_values):
        """Giai đoạn khai thác PUMA theo đúng paper"""
        Q = 0.67
        Beta = 2
        
        # Chuyển sang định dạng giống MATLAB để dễ thao tác
        Sol = [{'X': pop.copy(), 'Cost': fit} for pop, fit in zip(population, fitness_values)]
        NewSol = [{'X': {}, 'Cost': -np.inf} for _ in range(self.population_size)]
        
        # Tìm best solution (tối đa hóa)
        best_idx = np.argmax(fitness_values)
        Best = {'X': population[best_idx].copy(), 'Cost': fitness_values[best_idx]}
        
        # Tính mbest theo đúng công thức MATLAB
        mbest = {}
        for param in self.param_ranges.keys():
            if self.param_ranges[param]['type'] in ['int', 'float', 'log_uniform']:
                param_values = [s['X'][param] for s in Sol]
                mbest[param] = sum(param_values) / self.population_size
            else:
                # Cho categorical, lấy mode
                param_values = [s['X'][param] for s in Sol]
                mbest[param] = max(set(param_values), key=param_values.count)
        
        for i in range(self.population_size):
            new_individual = {}
            
            # Tạo random coefficients
            beta1 = 2 * random.random()
            
            for param, range_info in self.param_ranges.items():
                if range_info['type'] in ['int', 'float', 'log_uniform']:
                    # Tạo random vectors
                    beta2 = random.gauss(0, 1)
                    w = random.gauss(0, 1)  
                    v = random.gauss(0, 1)
                    
                    # Tính F1 và F2 (Eq 35, 36)
                    F1 = random.gauss(0, 1) * np.exp(2 - (i+1) * (2/self.generations))
                    F2 = w * (v**2) * np.cos((2 * random.random()) * w)
                    
                    # Tính R_1 (Eq 34)
                    R_1 = 2 * random.random() - 1
                    
                    # Tính S1, S2, VEC
                    S1 = 2 * random.random() - 1 + random.gauss(0, 1)
                    S2 = F1 * R_1 * Sol[i]['X'][param] + F2 * (1 - R_1) * Best['X'][param]
                    VEC = S2 / S1 if S1 != 0 else S2
                    
                    if random.random() <= 0.5:
                        if random.random() > Q:
                            # Eq 32 - phần đầu
                            random_idx = random.randint(0, self.population_size - 1)
                            new_val = (Best['X'][param] + 
                                    beta1 * np.exp(beta2) * 
                                    (Sol[random_idx]['X'][param] - Sol[i]['X'][param]))
                        else:
                            # Eq 32 - phần sau  
                            new_val = beta1 * VEC - Best['X'][param]
                    else:
                        # Eq 33
                        r1 = random.randint(0, self.population_size - 1)
                        sign = 1 if random.random() > 0.5 else -1
                        denominator = 1 + (Beta * random.random())
                        new_val = ((mbest[param] * Sol[r1]['X'][param] - 
                                sign * Sol[i]['X'][param]) / denominator)
                    
                    if range_info['type'] == 'int':
                        new_individual[param] = int(round(new_val))
                    else:
                        new_individual[param] = new_val
                        
                else:  # choice type
                    # Cho categorical parameters
                    if random.random() < 0.5:
                        new_individual[param] = Best['X'][param]
                    else:
                        new_individual[param] = random.choice(range_info['options'])
            
            # Clip parameters
            new_individual = self.clip_individual(new_individual)
            NewSol[i]['X'] = new_individual
            NewSol[i]['Cost'] = self.evaluate_individual(new_individual)
            
            # Cập nhật nếu tốt hơn (tối đa hóa) 
            if NewSol[i]['Cost'] > Sol[i]['Cost']:
                Sol[i] = NewSol[i].copy()
        
        # Convert back to lists
        new_population = [s['X'] for s in Sol]
        new_fitness = [s['Cost'] for s in Sol]
        
        return new_population, new_fitness
    
    def save_results_to_csv(self, filename=None):
        """Lưu kết quả từng vòng lặp ra file CSV"""
        if not self.iteration_results:
            print("Không có dữ liệu để lưu!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"puma_optimization_results_{timestamp}.csv"
        
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
            filename = f"puma_best_model_{timestamp}.joblib"  # Đổi extension
        
        # Sử dụng joblib thay vì pickle
        joblib.dump(self.best_model, filename)
        print(f"Mô hình tốt nhất đã được lưu vào: {filename}")
        
        return filename
    
    def calculate_experience_scores(self, iteration):
        """Tính toán scores cho Experience Management System"""
        
        # F1 calculations (Eq 1, 2, 13, 14)
        F1_Explor = self.PF[0] * (self.Seq_Cost_Explore[0] / self.Seq_Time_Explore[0])
        F1_Exploit = self.PF[0] * (self.Seq_Cost_Exploit[0] / self.Seq_Time_Exploit[0])
        
        # F2 calculations (Eq 3, 4, 15, 16)
        sum_cost_explore = sum(self.Seq_Cost_Explore)
        sum_time_explore = sum(self.Seq_Time_Explore) 
        sum_cost_exploit = sum(self.Seq_Cost_Exploit)
        sum_time_exploit = sum(self.Seq_Time_Exploit)
        
        F2_Explor = self.PF[1] * (sum_cost_explore / sum_time_explore)
        F2_Exploit = self.PF[1] * (sum_cost_exploit / sum_time_exploit)
        
        if iteration <= 3:
            # Unexperienced phase (Eq 11, 12)
            self.Score_Explore = (self.PF[0] * F1_Explor) + (self.PF[1] * F2_Explor)
            self.Score_Exploit = (self.PF[0] * F1_Exploit) + (self.PF[1] * F2_Exploit)
        else:
            # Experienced phase với adaptive mechanism (Eq 17, 18)
            if self.Score_Explore < self.Score_Exploit:
                self.Mega_Explor = max(self.Mega_Explor - 0.01, 0.01)
                self.Mega_Exploit = 0.99
            elif self.Score_Explore > self.Score_Exploit:
                self.Mega_Explor = 0.99
                self.Mega_Exploit = max(self.Mega_Exploit - 0.01, 0.01)
            
            lmn_Explore = 1 - self.Mega_Explor  # Eq 24
            lmn_Exploit = 1 - self.Mega_Exploit # Eq 22
            
            min_PF_F3 = min(self.PF_F3) if self.PF_F3 else 0.1
            
            # Eq 19, 20
            self.Score_Explore = (self.Mega_Explor * F1_Explor + 
                                self.Mega_Explor * F2_Explor + 
                                lmn_Explore * min_PF_F3 * self.F3_Explore)
            self.Score_Exploit = (self.Mega_Exploit * F1_Exploit + 
                                self.Mega_Exploit * F2_Exploit + 
                                lmn_Exploit * min_PF_F3 * self.F3_Exploit)
            
    def update_experience_data(self, iteration, phase_type, old_best_fitness, new_best_fitness):
        """Cập nhật dữ liệu kinh nghiệm sau mỗi phase"""
        
        cost_improvement = abs(old_best_fitness - new_best_fitness)
        
        if phase_type == 'exploration':
            # Shift arrays và cập nhật cost mới nhất
            self.Seq_Cost_Explore[2] = self.Seq_Cost_Explore[1] 
            self.Seq_Cost_Explore[1] = self.Seq_Cost_Explore[0]
            self.Seq_Cost_Explore[0] = cost_improvement
            
            if cost_improvement != 0:
                self.PF_F3.append(cost_improvement)
                
            self.F3_Explore = self.PF[2]
            self.F3_Exploit += self.PF[2]
            
            # Cập nhật counters
            self.UnSelected[1] += 1  # Exploit counter
            self.UnSelected[0] = 1   # Reset explore counter
            
        else:  # exploitation
            # Shift arrays và cập nhật cost mới nhất  
            self.Seq_Cost_Exploit[2] = self.Seq_Cost_Exploit[1]
            self.Seq_Cost_Exploit[1] = self.Seq_Cost_Exploit[0] 
            self.Seq_Cost_Exploit[0] = cost_improvement
            
            if cost_improvement != 0:
                self.PF_F3.append(cost_improvement)
                
            self.F3_Explore += self.PF[2]
            self.F3_Exploit = self.PF[2]
            
            # Cập nhật counters
            self.UnSelected[0] += 1  # Explore counter
            self.UnSelected[1] = 1   # Reset exploit counter
        
        # Cập nhật time sequences nếu có thay đổi flag
        if iteration >= 4 and self.Flag_Change != (1 if phase_type == 'exploration' else 2):
            self.Flag_Change = 1 if phase_type == 'exploration' else 2
            
            # Shift time arrays
            for i in [2, 1]:
                self.Seq_Time_Explore[i] = self.Seq_Time_Explore[i-1]
                self.Seq_Time_Exploit[i] = self.Seq_Time_Exploit[i-1]
            
            self.Seq_Time_Explore[0] = self.UnSelected[0]
            self.Seq_Time_Exploit[0] = self.UnSelected[1]
    
    def optimize(self, verbose=True, save_csv=True, save_model=True):
        """Chạy quá trình tối ưu hóa PUMA"""
        if self.evaluate_function is None:
            raise ValueError("Chưa thiết lập hàm đánh giá! Sử dụng set_evaluate_function()")
        
        # Khởi tạo quần thể
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Tìm nghiệm tốt nhất ban đầu
        best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[best_idx]
        best_individual = population[best_idx].copy()
        best_fitness = current_best_fitness
        
        # Cập nhật mô hình tốt nhất ban đầu
        try:
            self.best_model = self.create_model_from_params(best_individual)
            self.best_model.fit(self.X_train_scaled, self.y_train)
        except Exception as e:
            print(f"Lỗi khi tạo mô hình ban đầu: {e}")
        
        if verbose:
            print(f"\nTiến trình tối ưu hóa PUMA cho {self.model_type}:")
            print("Gen | Fitness     | R²        | MAE       | RMSE")
            print("-" * 55)
        
        # Vòng lặp tối ưu hóa
        # Vòng lặp tối ưu hóa
        for generation in range(self.generations):
            old_best_fitness = current_best_fitness
            
            if generation < 3:
                # Unexperienced Phase: chạy cả 2 phases để học
                
                # Chạy Exploration
                population_explore, fitness_explore = self.exploration_phase(population, fitness_values)
                best_explore_fitness = max(fitness_explore)
                self.Costs_Explor.append(best_explore_fitness)
                
                # Chạy Exploitation  
                population_exploit, fitness_exploit = self.exploitation_phase(population, fitness_values)
                best_exploit_fitness = max(fitness_exploit)
                self.Costs_Exploit.append(best_exploit_fitness)
                
                # Combine và chọn best solutions
                all_population = population + population_explore + population_exploit
                all_fitness = fitness_values + fitness_explore + fitness_exploit
                
                # Sắp xếp và chọn top population_size
                sorted_indices = np.argsort(all_fitness)[::-1]  # Descending
                population = [all_population[i] for i in sorted_indices[:self.population_size]]
                fitness_values = [all_fitness[i] for i in sorted_indices[:self.population_size]]
                
                current_best_fitness = fitness_values[0]
                best_individual = population[0].copy()
                
                # Khởi tạo experience data cho 3 vòng đầu
                if generation == 0:
                    self.Initial_Best = best_individual.copy()
                    initial_fitness = current_best_fitness
                
                # Cập nhật Seq_Cost theo equations (5)-(10) 
                if generation == 0:
                    self.Seq_Cost_Explore[0] = abs(initial_fitness - self.Costs_Explor[0])
                    self.Seq_Cost_Exploit[0] = abs(initial_fitness - self.Costs_Exploit[0])
                elif generation == 1:  
                    self.Seq_Cost_Explore[1] = abs(self.Costs_Explor[1] - self.Costs_Explor[0])
                    self.Seq_Cost_Exploit[1] = abs(self.Costs_Exploit[1] - self.Costs_Exploit[0])
                elif generation == 2:
                    self.Seq_Cost_Explore[2] = abs(self.Costs_Explor[2] - self.Costs_Explor[1]) 
                    self.Seq_Cost_Exploit[2] = abs(self.Costs_Exploit[2] - self.Costs_Exploit[1])
                    
                    # Khởi tạo PF_F3
                    for cost_val in self.Seq_Cost_Explore + self.Seq_Cost_Exploit:
                        if cost_val != 0:
                            self.PF_F3.append(cost_val)
                
                # Tính experience scores
                self.calculate_experience_scores(generation + 1)
                
                # Cập nhật best_fitness nếu tốt hơn
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    
                # Cập nhật mô hình tốt nhất
                try:
                    self.best_model = self.create_model_from_params(best_individual)
                    self.best_model.fit(self.X_train_scaled, self.y_train)
                except Exception as e:
                    print(f"Lỗi khi cập nhật mô hình tốt nhất: {e}")
                
            else:
                # Experienced Phase: chọn phase dựa trên scores
                if not self.experienced_phase_started:
                    self.experienced_phase_started = True
                    
                if self.Score_Explore > self.Score_Exploit:
                    # Chọn Exploration
                    population, fitness_values = self.exploration_phase(population, fitness_values)
                    self.update_experience_data(generation + 1, 'exploration', 
                                            old_best_fitness, max(fitness_values))
                else:
                    # Chọn Exploitation
                    population, fitness_values = self.exploitation_phase(population, fitness_values)
                    self.update_experience_data(generation + 1, 'exploitation',
                                            old_best_fitness, max(fitness_values))
                
                current_best_fitness = max(fitness_values)
                best_idx = np.argmax(fitness_values)
                
                if current_best_fitness > best_fitness:
                    best_individual = population[best_idx].copy() 
                    best_fitness = current_best_fitness
                    
                    # Cập nhật mô hình tốt nhất
                    try:
                        self.best_model = self.create_model_from_params(best_individual)
                        self.best_model.fit(self.X_train_scaled, self.y_train)
                    except Exception as e:
                        print(f"Lỗi khi cập nhật mô hình tốt nhất: {e}")
                
                # Tính lại experience scores cho iteration tiếp theo
                self.calculate_experience_scores(generation + 1)
            
            self.best_scores_history.append(current_best_fitness)
            
            # Tính toán metrics chi tiết
            if self.model_type == 'regression':
                r2, mae, rmse = self.calculate_metrics(best_individual)
                
                # Lưu kết quả vòng lặp
                iteration_result = {
                    'generation': generation + 1,
                    'fitness': current_best_fitness,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Thêm tham số tốt nhất vào kết quả
                for param_name, value in best_individual.items():
                    iteration_result[f'best_{param_name}'] = value
                
                self.iteration_results.append(iteration_result)
                
                if verbose and generation % 1 == 0:  # In mỗi 1 generation
                    print(f"{generation+1:3d} | {current_best_fitness:10.6f} | {r2:8.6f} | {mae:8.4f} | {rmse:8.4f}")
        
        # In kết quả cuối cùng
        if verbose:
            final_r2, final_mae, final_rmse = self.calculate_metrics(best_individual)
            print(f"\nKết quả cuối cùng:")
            print(f"{self.generations:3d} | {best_fitness:10.6f} | {final_r2:8.6f} | {final_mae:8.4f} | {final_rmse:8.4f}")
        
        # Lưu trữ kết quả cuối cùng
        self.best_individual = best_individual
        self.best_score = best_fitness
        
        # Lưu file nếu được yêu cầu
        csv_filename = None
        model_filename = None
        
        if save_csv:
            csv_filename = self.save_results_to_csv()
        
        if save_model:
            model_filename = self.save_best_model()
        
        return {
            'best_params': self.best_individual, 
            'best_score': self.best_score,
            'best_model': self.best_model,
            'csv_file': csv_filename,
            'model_file': model_filename
        }
    
    def plot_optimization_progress(self, title="Tiến trình tối ưu hóa PUMA"):
        """Vẽ biểu đồ tiến trình tối ưu hóa"""
        if not self.best_scores_history:
            print("Chưa có dữ liệu tối ưu hóa để vẽ biểu đồ!")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_scores_history, 'b-', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Thế hệ', fontsize=12)
        plt.ylabel('Điểm số tốt nhất', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_results_dataframe(self):
        """Trả về DataFrame chứa kết quả tối ưu hóa"""
        if not self.iteration_results:
            print("Chưa có dữ liệu tối ưu hóa!")
            return None
        
        return pd.DataFrame(self.iteration_results)