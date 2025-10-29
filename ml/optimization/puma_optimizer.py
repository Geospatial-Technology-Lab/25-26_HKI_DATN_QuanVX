import numpy as np
import random
from sklearn.model_selection import train_test_split

from config.model_params import RF_PARAM_RANGES, XGB_PARAM_RANGES, SVM_PARAM_RANGES
from evaluation.evaluation_utils import evaluate_regression_model, load_data_from_csv
from optimization.model_utils import generate_random_params, create_model

RANDOM_SEED = 42

class PUMAOptimizer:
    def __init__(self, X=None, y=None, model_type='regression', population_size=10, generations=100, random_state=None):
        self.model_type = model_type
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state or RANDOM_SEED
        self.best_individual = None
        self.best_score = -np.inf
        self.best_scores_history = []
        self.pCR = 0.5
        self.p = 0.1
        self.best_model = None
        self.iteration_results = []
        self.UnSelected = [1, 1]
        self.F3_Explore = 0
        self.F3_Exploit = 0
        self.Seq_Time_Explore = [1, 1, 1]
        self.Seq_Time_Exploit = [1, 1, 1]
        self.Seq_Cost_Explore = [1, 1, 1]  
        self.Seq_Cost_Exploit = [1, 1, 1]
        self.Score_Explore = 0
        self.Score_Exploit = 0
        self.PF = [0.5, 0.5, 0.3]
        self.PF_F3 = []
        self.Mega_Explor = 0.99
        self.Mega_Exploit = 0.99
        self.Flag_Change = 1
        self.Initial_Best = None
        self.Costs_Explor = []
        self.Costs_Exploit = []
        self.experienced_phase_started = False
        
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        if X is not None and y is not None:
            X_prepared, y_prepared = X, y
            self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            # Sử dụng load_data_from_csv từ evaluation_utils
            # Cần truyền csv_file_path từ bên ngoài
            raise ValueError("Cần cung cấp X và y hoặc sử dụng load_data_from_csv từ bên ngoài")

        stratify = y_prepared if model_type == 'classification' else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_prepared, y_prepared, 
            test_size=0.2,
            stratify=stratify, 
            random_state=self.random_state
        )

        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        self.param_ranges = {}
    
    def set_param_ranges(self, param_ranges):
        self.param_ranges = param_ranges
    
    def set_param_ranges_by_model_type(self, model_type):
        """Tự động set param ranges dựa trên model type"""
        model_type = model_type.lower()
        if 'rf' in model_type or 'random_forest' in model_type:
            self.param_ranges = RF_PARAM_RANGES
        elif 'xgb' in model_type or 'xgboost' in model_type:
            self.param_ranges = XGB_PARAM_RANGES
        elif 'svm' in model_type or 'support_vector' in model_type:
            self.param_ranges = SVM_PARAM_RANGES
        else:
            raise ValueError(f"Model type '{model_type}' không được hỗ trợ. Chỉ hỗ trợ: rf, xgb, svm")
    
    def load_data_from_file(self, csv_file_path, test_size=0.2):
        """Tiện ích để load data từ CSV file"""
        X_train, X_test, y_train, y_test = load_data_from_csv(csv_file_path, test_size, self.random_state)
        self.X_train = X_train
        self.X_test = X_test  
        self.y_train = y_train
        self.y_test = y_test
        self.feature_columns = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()

    def create_individual(self):
        return generate_random_params(self.param_ranges)

    def create_model_from_params(self, individual):
        if 'n_estimators' in individual:
            model_type = 'xgb' if 'learning_rate' in individual else 'rf'
        elif 'C' in individual and 'gamma' in individual:
            model_type = 'svm'
        else:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        
        return create_model(model_type, individual, self.random_state)
    
    def calculate_metrics(self, individual):
        if self.model_type != 'regression':
            return None, None, None
        
        try:
            model = self.create_model_from_params(individual)
            
            detailed_metrics = evaluate_regression_model(
                model, self.X_train_scaled, self.X_test_scaled, 
                self.y_train, self.y_test, 
                clip_predictions=True, return_detailed=True
            )
            
            return detailed_metrics['r2'], detailed_metrics['mae'], detailed_metrics['rmse']
            
        except Exception as e:
            return 0.0, float('inf'), float('inf')
    
    def clip_individual(self, individual):
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
    
    def _evaluate_individual(self, individual):
        """Evaluate fitness của một cá thể"""
        try:
            model = self.create_model_from_params(individual)
            return evaluate_regression_model(
                model, self.X_train_scaled, self.X_test_scaled,
                self.y_train, self.y_test, clip_predictions=True, return_detailed=False
            )
        except Exception:
            return -np.inf
    
    def exploration_phase(self, population, fitness_values):
        sorted_indices = np.argsort(fitness_values)[::-1]
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
            new_fitness_val = self._evaluate_individual(new_individual)
            
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
            NewSol[i]['Cost'] = self._evaluate_individual(new_individual)
            
            # Cập nhật nếu tốt hơn (tối đa hóa) 
            if NewSol[i]['Cost'] > Sol[i]['Cost']:
                Sol[i] = NewSol[i].copy()
        
        # Convert back to lists
        new_population = [s['X'] for s in Sol]
        new_fitness = [s['Cost'] for s in Sol]
        
        return new_population, new_fitness
    
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
    
    def optimize(self, verbose=False):
        """Chạy quá trình tối ưu hóa PUMA"""
        if not self.param_ranges:
            raise ValueError("Chưa thiết lập param ranges! Sử dụng set_param_ranges hoặc set_param_ranges_by_model_type")
        
        if verbose:
            print(f"{'Vong':>5} {'Fitness':>12} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
            print("-" * 52)
        
        # Khởi tạo quần thể
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self._evaluate_individual(ind) for ind in population]
        
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
            pass
        
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
                    pass
                
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
                        pass
                
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
                
                if verbose:
                    print(f"{generation+1:5d} {current_best_fitness:12.6f} {r2:10.6f} {mae:10.6f} {rmse:10.6f}")
        
        # Lưu trữ kết quả cuối cùng
        self.best_individual = best_individual
        self.best_score = best_fitness
        
        return {
            'best_params': self.best_individual, 
            'best_score': self.best_score,
            'best_model': self.best_model
        }