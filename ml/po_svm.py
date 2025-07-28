import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=20, generations=20):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.best_scores_history = []  # Theo dõi điểm số tốt nhất để vẽ biểu đồ
        
        # Chia dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Chuẩn hóa dữ liệu (rất quan trọng cho SVM)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Phạm vi tham số SVM - Bộ tham số mở rộng
        self.param_ranges = {
            'C': {'type': 'log_uniform', 'min': 0.001, 'max': 1000},
            'gamma': {'type': 'log_uniform', 'min': 0.0001, 'max': 10},
            'kernel': {'type': 'choice', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
            'degree': {'type': 'int', 'min': 2, 'max': 5},
            'coef0': {'type': 'float', 'min': 0.0, 'max': 10.0},
            'tol': {'type': 'log_uniform', 'min': 1e-5, 'max': 1e-2},
            'class_weight': {'type': 'choice', 'options': [None, 'balanced']},
            'max_iter': {'type': 'int', 'min': 1000, 'max': 10000},
            'shrinking': {'type': 'choice', 'options': [True, False]},
            'probability': {'type': 'choice', 'options': [True, False]}
        }
        
        # Lấy tham số số học để thực hiện các phép toán vector nhất quán
        self.numerical_params = [p for p in self.param_ranges if self.param_ranges[p]['type'] in ['float', 'int']]
        self.categorical_params = [p for p in self.param_ranges if self.param_ranges[p]['type'] == 'categorical']
        self.num_numerical = len(self.numerical_params)
        
        # Tham số đặc trưng PUMA
        self.PF = [0.5, 0.5, 0.3]  # Tham số cho F1, F2, F3
        self.unselected = [1, 1]  # [Khám phá, Khai thác]
        self.F3_explore = 0
        self.F3_exploit = 0
        self.seq_time_explore = [1, 1, 1]
        self.seq_time_exploit = [1, 1, 1]
        self.seq_cost_explore = [0, 0, 0]
        self.seq_cost_exploit = [0, 0, 0]
        self.score_explore = 0
        self.score_exploit = 0
        self.PF_F3 = []
        self.mega_explore = 0.99
        self.mega_exploit = 0.99
    
    def create_individual(self):
        """Tạo một cá thể ngẫu nhiên (bộ tham số) cho SVM"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):  # Phạm vi liên tục
                if range_info['type'] == 'int':
                    individual[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
                elif range_info['type'] == 'float':
                    if param in ['C', 'gamma']:
                        # Lấy mẫu theo thang log cho C và gamma
                        log_min = np.log10(range_info['min'])
                        log_max = np.log10(range_info['max'])
                        individual[param] = 10 ** np.random.uniform(log_min, log_max)
                    else:
                        individual[param] = np.random.uniform(range_info['min'], range_info['max'])
            else:  # Lựa chọn rời rạc (như kernel)
                individual[param] = np.random.choice(range_info)
        return individual
    
    def evaluate_individual(self, individual):
        """Đánh giá tham số SVM của cá thể"""
        try:
            svm_params = {
                'C': individual['C'],
                'kernel': individual['kernel'],
                'tol': individual['tol'],
                'random_state': 42,
                'class_weight': 'balanced',
                'probability': True
            }
            
            if individual['kernel'] == 'rbf':
                svm_params['gamma'] = individual['gamma']
            elif individual['kernel'] == 'poly':
                svm_params['gamma'] = individual['gamma']
                svm_params['degree'] = individual['degree']
                svm_params['coef0'] = individual['coef0']
            elif individual['kernel'] == 'sigmoid':
                svm_params['gamma'] = individual['gamma']
                svm_params['coef0'] = individual['coef0']
            
            svm = SVC(**svm_params)
            cv_scores = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=3, scoring='f1')
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Lỗi đánh giá cá thể: {str(e)}")
            return -np.inf
    
    def exploration_phase(self, population):
        """Giai đoạn khám phá PUMA"""
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Chọn 6 giải pháp khác nhau một cách ngẫu nhiên
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Tạo giải pháp mới
            new_individual = {}
            for param, range_info in self.param_ranges.items():
                if isinstance(range_info, dict):  # Tham số liên tục
                    if random.random() < 0.5:
                        if range_info['type'] == 'int':
                            new_individual[param] = random.randint(range_info['min'], range_info['max'])
                        else:
                            new_individual[param] = random.uniform(range_info['min'], range_info['max'])
                    else:
                        G = 2 * random.random() - 1
                        term1 = a[param] + G * (a[param] - b[param])
                        term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                   ((c[param] - d[param]) - (e[param] - f[param])))
                        new_val = term1 + term2
                        
                        if range_info['type'] == 'int':
                            new_val = int(round(np.clip(new_val, range_info['min'], range_info['max'])))
                        else:
                            new_val = np.clip(new_val, range_info['min'], range_info['max'])
                        new_individual[param] = new_val
                else:  # Tham số phân loại (như kernel)
                    new_individual[param] = random.choice(range_info)
            
            # Đánh giá và cập nhật
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > self.evaluate_individual(current):
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(self.evaluate_individual(current))
        
        return new_population, new_fitness

    def exploitation_phase(self, population, best_solution, iteration, max_iter):
        """Giai đoạn khai thác PUMA"""
        Q = 0.67  # Hằng số khai thác
        Beta = 2  # Hằng số Beta
        new_population = []
        new_fitness = []
        
        # Lấy giải pháp tốt nhất
        best_idx = np.argmax([self.evaluate_individual(ind) for ind in population])
        best_solution = population[best_idx]
        
        # Tính vị trí trung bình
        mbest = {}
        for param in self.param_ranges:
            if isinstance(self.param_ranges[param], dict):
                mbest[param] = np.mean([p[param] for p in population])
            else:  # Tham số phân loại
                values = [p[param] for p in population]
                mbest[param] = max(set(values), key=values.count)
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            # Handle categorical parameters separately
            for param in self.param_ranges:
                if not isinstance(self.param_ranges[param], dict):  # categorical
                    if random.random() < 0.5:
                        new_individual[param] = random.choice(self.param_ranges[param]['values'])
                    else:
                        new_individual[param] = best_solution[param]
            
            # Handle numerical parameters
            num_params = len([p for p in self.param_ranges.items() if isinstance(p[1], dict)])
            if num_params > 0:
                beta1 = 2 * random.random()
                beta2 = np.random.randn(num_params)
                w = np.random.randn(num_params)
                v = np.random.randn(num_params)
                F1 = np.random.randn(num_params) * np.exp(2 - iteration * (2/max_iter))
                F2 = w * np.power(v, 2) * np.cos((2 * random.random()) * w)
                R_1 = 2 * random.random() - 1
                
                if random.random() <= 0.5:
                    S1 = 2 * random.random() - 1 + np.random.randn(num_params)
                    
                    # Process numerical parameters
                    current_array = []
                    best_array = []
                    param_names = []
                    
                    for param, range_info in self.param_ranges.items():
                        if isinstance(range_info, dict):
                            current_array.append(float(current[param]))
                            best_array.append(float(best_solution[param]))
                            param_names.append(param)
                    
                    current_array = np.array(current_array)
                    best_array = np.array(best_array)
                    
                    S2 = F1 * R_1 * current_array + F2 * (1 - R_1) * best_array
                    VEC = np.divide(S2, S1, out=np.zeros_like(S2), where=S1!=0)
                    
                    if random.random() > Q:
                        random_sol = random.choice(population)
                        random_array = np.array([float(random_sol[param]) for param in param_names])
                        new_pos = best_array + beta1 * (np.exp(beta2)) * (random_array - current_array)
                    else:
                        new_pos = beta1 * VEC - best_array
                else:
                    r1 = random.randint(0, self.population_size-1)
                    r1_sol = population[r1]
                    
                    current_array = []
                    r1_array = []
                    mbest_array = []
                    param_names = []
                    
                    for param, range_info in self.param_ranges.items():
                        if isinstance(range_info, dict):
                            current_array.append(float(current[param]))
                            r1_array.append(float(r1_sol[param]))
                            mbest_array.append(float(mbest[param]))
                            param_names.append(param)
                    
                    current_array = np.array(current_array)
                    r1_array = np.array(r1_array)
                    mbest_array = np.array(mbest_array)
                    
                    sign = 1 if random.random() > 0.5 else -1
                    new_pos = (mbest_array * r1_array - sign * current_array) / (1 + (Beta * random.random()))
                
                # Convert numerical results back to parameters
                for j, param in enumerate(param_names):
                    range_info = self.param_ranges[param]
                    if range_info['type'] == 'int':
                        new_individual[param] = int(round(np.clip(new_pos[j], 
                                                                range_info['min'], 
                                                                range_info['max'])))
                    else:
                        new_individual[param] = np.clip(new_pos[j], 
                                                      range_info['min'], 
                                                      range_info['max'])
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > self.evaluate_individual(current):
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(self.evaluate_individual(current))
        
        return new_population, new_fitness

    def optimize(self):
        """Main PUMA optimization algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Khởi tạo theo dõi kết quả
        iteration_results = []
        
        # Tốt nhất ban đầu
        best_idx = np.argmax(fitness_values)
        self.best_individual = population[best_idx]
        self.best_score = fitness_values[best_idx]
        initial_best_score = self.best_score
        self.best_scores_history.append(self.best_score)
        
        # Tham số cho việc chọn phase
        unselected = [1, 1]  # [Khám phá, Khai thác]
        seq_time_explore = [1, 1, 1]
        seq_time_exploit = [1, 1, 1]
        seq_cost_explore = [0.1, 0.1, 0.1]
        seq_cost_exploit = [0.1, 0.1, 0.1]
        pf = [0.5, 0.5, 0.3]  # Trọng số cho F1, F2, F3
        mega_explor = 0.99
        mega_exploit = 0.99
        f3_explore = 0
        f3_exploit = 0
        pf_f3 = [0.01]
        flag_change = 1
        
        # Giai đoạn chưa có kinh nghiệm (3 iterations đầu tiên)
        for iteration in range(3):
            print(f"Iteration {iteration + 1}/3")
            
            # Exploration
            pop_explore, fit_explore = self.exploration_phase(population)
            cost_explore = max(fit_explore)
            
            # Exploitation
            pop_exploit, fit_exploit = self.exploitation_phase(population, self.best_individual, iteration, self.generations)
            cost_exploit = max(fit_exploit)
            
            # Combine and select best solutions
            population = population + pop_explore + pop_exploit
            fitness_values = fitness_values + fit_explore + fit_exploit
            indices = np.argsort(fitness_values)[::-1][:self.population_size]
            population = [population[i] for i in indices]
            fitness_values = [fitness_values[i] for i in indices]
            
            # Cập nhật tốt nhất
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0]
                print(f"Điểm số tốt nhất mới: {self.best_score:.4f}")
                self.best_scores_history.append(self.best_score)
        
        # Khởi tạo chi phí sequence
        seq_cost_explore[0] = max(0.01, abs(initial_best_score - cost_explore))
        seq_cost_exploit[0] = max(0.01, abs(initial_best_score - cost_exploit))
        
        # Thêm chi phí khác không vào PF_F3
        for cost in seq_cost_explore + seq_cost_exploit:
            if cost > 0.01:
                pf_f3.append(cost)
        
        # Tính điểm số ban đầu
        f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
        f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
        f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
        f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
        score_explore = pf[0] * f1_explore + pf[1] * f2_explore
        score_exploit = pf[0] * f1_exploit + pf[1] * f2_exploit
        
        # Experienced Phase
        for iteration in range(3, self.generations):
            # Early stopping: dừng sớm nếu không có cải thiện sau 5 iterations
            if len(self.best_scores_history) > 5:
                if all(score == self.best_scores_history[-1] for score in self.best_scores_history[-5:]):
                    print(f"Early stopping tại iteration {iteration + 1}: Không có cải thiện trong 5 iterations liên tiếp")
                    break
                    
            print(f"Iteration {iteration + 1}/{self.generations}")
            
            if score_explore > score_exploit:
                # Khám phá
                population, fitness_values = self.exploration_phase(population)
                count_select = [unselected[0], unselected[1]]
                unselected[1] += 1
                unselected[0] = 1
                f3_explore = pf[2]
                f3_exploit += pf[2]
                
                # Cập nhật chi phí sequence
                if fitness_values[0] > self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_explore = [max(0.01, cost_diff)] + seq_cost_explore[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            else:
                # Khai thác
                population, fitness_values = self.exploitation_phase(population, self.best_individual, iteration, self.generations)
                count_select = [unselected[0], unselected[1]]
                unselected[0] += 1
                unselected[1] = 1
                f3_explore += pf[2]
                f3_exploit = pf[2]
                
                # Cập nhật chi phí sequence
                if fitness_values[0] > self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_exploit = [max(0.01, cost_diff)] + seq_cost_exploit[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            
            # Cập nhật giải pháp tốt nhất
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0]
                print(f"Điểm số tốt nhất mới: {self.best_score:.4f}")
                self.best_scores_history.append(self.best_score)
            
            # Cập nhật time sequences nếu phase thay đổi
            if flag_change != (1 if score_explore > score_exploit else 2):
                flag_change = 1 if score_explore > score_exploit else 2
                seq_time_explore = [count_select[0]] + seq_time_explore[:2]
                seq_time_exploit = [count_select[1]] + seq_time_exploit[:2]
            
            # Update scores
            if score_explore < score_exploit:
                mega_explor = max(mega_explor - 0.01, 0.01)
                mega_exploit = 0.99
            elif score_explore > score_exploit:
                mega_explor = 0.99
                mega_exploit = max(mega_exploit - 0.01, 0.01)
            
            lmn_explore = 1 - mega_explor
            lmn_exploit = 1 - mega_exploit
            
            f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
            f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
            f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
            f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
            
            min_pf_f3 = min(pf_f3) if pf_f3 else 0.01
            score_explore = (mega_explor * f1_explore) + (mega_explor * f2_explore) + (lmn_explore * min_pf_f3 * f3_explore)
            score_exploit = (mega_exploit * f1_exploit) + (mega_exploit * f2_exploit) + (lmn_exploit * min_pf_f3 * f3_exploit)
        
        # Lưu kết quả iteration (đơn giản hóa)
        for iteration in range(self.generations):
            iteration_results.append({
                'iteration': iteration + 1,
                'best_score': self.best_score
            })
        
        return self.best_individual, self.best_score, iteration_results

    def evaluate_final_model(self):
        """Đánh giá mô hình SVM cuối cùng trên tập test"""
        if self.best_individual is None:
            print("Không có mô hình được tối ưu!")
            return None
        
        # Tạo SVM với tham số tốt nhất
        svm_params = {
            'C': self.best_individual['C'],
            'kernel': self.best_individual['kernel'],
            'tol': self.best_individual['tol'],
            'random_state': 42,
            'class_weight': 'balanced',
            'probability': True
        }
        
        # Thêm tham số đặc trưng cho kernel
        if self.best_individual['kernel'] == 'rbf':
            svm_params['gamma'] = self.best_individual['gamma']
        elif self.best_individual['kernel'] == 'poly':
            svm_params['gamma'] = self.best_individual['gamma']
            svm_params['degree'] = self.best_individual['degree']
            svm_params['coef0'] = self.best_individual['coef0']
        elif self.best_individual['kernel'] == 'sigmoid':
            svm_params['gamma'] = self.best_individual['gamma']
            svm_params['coef0'] = self.best_individual['coef0']
        
        best_svm = SVC(**svm_params)
        best_svm.fit(self.X_train_scaled, self.y_train)
        
        # Dự đoán trên tập test
        y_pred = best_svm.predict(self.X_test_scaled)
        y_prob = best_svm.predict_proba(self.X_test_scaled)
        
        # Lấy xác suất cho lớp 1
        if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        # Tính các metric
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print("\nCác metric trên tập test:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        # Lấy tên feature
        feature_names = getattr(self, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        # Với SVM, không thể lấy trực tiếp feature importances như RF
        # Nhưng có thể lấy thông tin support vectors
        support_vector_info = {
            'n_support_vectors': best_svm.n_support_,
            'support_vectors_indices': best_svm.support_,
            'dual_coef': best_svm.dual_coef_
        }
        
        return {
            'model': best_svm,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_individual,
            'support_vector_info': support_vector_info
        }

def main():
    """Hàm chính cho tối ưu SVM"""
    print("Đọc dữ liệu từ file Excel...")
    
    # Thay đổi đường dẫn này thành file dữ liệu của bạn
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Đọc {len(df)} dòng dữ liệu")
        
        # Cột feature (điều chỉnh theo file Excel của bạn)
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Cột label (điều chỉnh theo file Excel của bạn)
        label_column = 'label_column'  # 1 = lũ lụt, 0 = không lũ lụt
        
        # Kiểm tra cột bị thiếu
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            print(f"CẢNH BÁO: Không tìm thấy các cột sau: {missing_cols}")
            print(f"Các cột có sẵn: {list(df.columns)}")
            return
        
        # Chuẩn bị dữ liệu
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Xử lý giá trị thiếu
        if np.isnan(X).any():
            print("CẢNH BÁO: Tìm thấy giá trị thiếu trong dữ liệu!")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        print(f"Kích thước features: {X.shape}")
        print("Phân bố label:")
        y_array = np.asarray(y, dtype=int)
        unique_labels = np.unique(y_array)
        label_counts = np.bincount(y_array)
        for label, count in zip(unique_labels, label_counts):
            print(f"  Lớp {label}: {count}")
        
        # Khởi tạo và chạy PUMA optimizer cho SVM
        print("Bắt đầu tối ưu PUMA...")
        optimizer = PUMAOptimizer(X, y, population_size=15, generations=10)
        best_params, best_score, iteration_results = optimizer.optimize()
        
        print("\nTối ưu hoàn thành!")
        print(f"Điểm F1 tốt nhất: {best_score:.4f}")
        print("\nTham số tốt nhất:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Huấn luyện mô hình cuối cùng với tham số tốt nhất
        final_model = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params.get('gamma', 'scale'),
            random_state=42
        )
        
        # Huấn luyện và đánh giá trên tập test
        final_model.fit(optimizer.X_train_scaled, optimizer.y_train)
        y_pred = final_model.predict(optimizer.X_test_scaled)
        
        # Tính các metric đơn giản
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(optimizer.y_test, y_pred)
        f1 = f1_score(optimizer.y_test, y_pred)
        
        print(f"\nKết quả cuối cùng:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
            
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        print("Vui lòng đảm bảo file Excel tồn tại tại đường dẫn đã chỉ định")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()