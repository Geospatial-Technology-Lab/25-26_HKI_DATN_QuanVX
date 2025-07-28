import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = np.inf  # RMSE: giá trị thấp hơn là tốt hơn
        self.best_scores_history = []  # Theo dõi các điểm số tốt nhất để vẽ biểu đồ
        
        # Chia tách và chuẩn hóa dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=None  # Loại bỏ stratify cho bài toán hồi quy
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Các tham số cần tối ưu hóa - Bộ tham số mở rộng
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
            'max_depth': {'type': 'int', 'min': 3, 'max': 15},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
            'subsample': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'colsample_bylevel': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'colsample_bynode': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 10},
            'gamma': {'type': 'float', 'min': 0.0, 'max': 5.0},
            'max_delta_step': {'type': 'int', 'min': 0, 'max': 10},
            'scale_pos_weight': {'type': 'float', 'min': 0.5, 'max': 2.0}
        }
    
    #Hàm tạo cá thể
    def create_individual(self):
        """Tạo một cá thể ngẫu nhiên (bộ tham số)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = random.randint(range_info['min'], range_info['max'])
            else:
                individual[param] = random.uniform(range_info['min'], range_info['max'])
        return individual

    #Hàm đánh giá cá thể, sử dụng để tính hàm fitness
    def evaluate_individual(self, individual):
        """Đánh giá fitness của một cá thể sử dụng RMSE (giá trị thấp hơn là tốt hơn)"""
        try:
            model = xgb.XGBRegressor(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                learning_rate=individual['learning_rate'],
                subsample=individual['subsample'],
                colsample_bytree=individual['colsample_bytree'],
                min_child_weight=individual['min_child_weight'],
                gamma=individual['gamma'],
                random_state=42,
                verbosity=0
            )
            
            # Sử dụng chia tách train-validation để đánh giá nhất quán
            X_train_val, X_val, y_train_val, y_val = train_test_split(
                self.X_train_scaled, self.y_train, test_size=0.2, random_state=42
            )
            
            model.fit(X_train_val, y_train_val)
            y_pred = model.predict(X_val)
            
            # Tính toán RMSE
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Lưu trữ các chỉ số trong cá thể
            individual['_metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            return rmse  # Trả về RMSE (giá trị thấp hơn là tốt hơn)
        except:
            individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
            return np.inf

    def exploration_phase(self, population, fitness_values):
        """Giai đoạn Khám phá PUMA cho bài toán hồi quy"""
        new_population = []
        new_fitness = []
        used_combinations = set()  # Giúp tránh trùng lặp Solutions
        pCR = 0.5  # Xác suất lai ghép
        p = 0.1    # Giá trị gia tăng để điều chỉnh pCR
        
        for i in range(self.population_size):
            current = population[i]
            
            # Chọn 6 nghiệm khác nhau một cách ngẫu nhiên
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            
            # Xử lý trường hợp kích thước quần thể quá nhỏ
            if len(available_indices) < 6:
                # Nếu không đủ cá thể, lặp lại một số chỉ số với thay thế
                selected_indices = random.choices(available_indices, k=6)
            else:
                selected_indices = random.sample(available_indices, 6)
            
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Tạo nghiệm mới
            attempts = 0
            max_attempts = 10  # Giới hạn số lần thử để tránh vòng lặp vô tận
            while attempts < max_attempts:
                new_individual = current.copy()  # Bắt đầu với nghiệm hiện tại
                
                # Đảm bảo ít nhất một tham số thay đổi bằng cách chọn tham số ngẫu nhiên
                j0 = random.choice(list(self.param_ranges.keys()))
                
                for param, range_info in self.param_ranges.items():
                    # Luôn thay đổi tham số j0 hoặc dựa trên xác suất pCR
                    if param == j0 or random.random() <= pCR:
                        if random.random() < 0.5:
                            # Tạo giá trị ngẫu nhiên với nhiễu bổ sung để đa dạng hóa tốt hơn
                            range_size = range_info['max'] - range_info['min']
                            noise = random.gauss(0, range_size * 0.1)
                            rand_val = random.random() * range_size + range_info['min'] + noise
                            rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                            new_individual[param] = self._apply_bounds_and_type(param, rand_val)
                        else:
                            # Phương trình khám phá PUMA với các phép toán vector thích hợp
                            G = 2 * random.random() - 1
                            term1 = a[param] + G * (a[param] - b[param])
                            term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                       ((c[param] - d[param]) - (e[param] - f[param])))
                            new_val = term1 + term2
                            new_individual[param] = self._apply_bounds_and_type(param, new_val)
                
                # Kiểm tra xem tổ hợp này có độc nhất không
                # Chỉ sử dụng giá trị tham số, loại trừ _metrics
                param_values = tuple(v for k, v in new_individual.items() if k != '_metrics')
                if param_values not in used_combinations:
                    used_combinations.add(param_values)
                    break
                attempts += 1
            
            # Đánh giá và cập nhật (RMSE: giá trị thấp hơn là tốt hơn)
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val < fitness_values[i]:  # RMSE thấp hơn là tốt hơn
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Cập nhật pCR khi không có cải thiện
                pCR = min(0.9, pCR + p)
        
        return new_population, new_fitness

    def _apply_bounds_and_type(self, param, new_val):
        """Áp dụng giới hạn và xử lý kiểu dữ liệu cho tham số"""
        range_info = self.param_ranges[param]
        clipped_val = np.clip(new_val, range_info['min'], range_info['max'])
        
        if range_info['type'] == 'int':
            return int(round(clipped_val))
        else:
            return round(clipped_val, 6)

    def exploitation_phase(self, population, fitness_values):
        """Giai đoạn Khai thác PUMA cho bài toán hồi quy"""
        Q = 0.67  # Hằng số khai thác
        Beta = 2  # Hằng số Beta
        new_population = []
        new_fitness = []
        
        # Lấy nghiệm tốt nhất (RMSE: giá trị thấp hơn là tốt hơn)
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        
        # Tính toán vị trí trung bình
        mbest = {}
        for param in self.param_ranges:
            if self.param_ranges[param]['type'] == 'int':
                mbest[param] = int(np.mean([p[param] for p in population]))
            else:
                mbest[param] = np.mean([p[param] for p in population])
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            beta1 = 2 * random.random()
            beta2 = np.random.randn(len(self.param_ranges))
            
            w = np.random.randn(len(self.param_ranges))  # Phương trình 37
            v = np.random.randn(len(self.param_ranges))  # Phương trình 38
            
            # Phương trình 35
            F1 = np.random.randn(len(self.param_ranges)) * np.exp(2 - i * (2/self.generations))
            # Phương trình 36
            F2 = w * np.power(v, 2) * np.cos((2 * random.random()) * w)
            
            R_1 = 2 * random.random() - 1  # Phương trình 34
            
            if random.random() <= 0.5:
                # Tính toán S1 và S2
                S1 = 2 * random.random() - 1 + np.random.randn(len(self.param_ranges))
                
                # Chuyển đổi thành mảng cho các phép toán vector
                current_array = np.array([current[param] for param in self.param_ranges])
                best_array = np.array([best_solution[param] for param in self.param_ranges])
                
                S2 = F1 * R_1 * current_array + F2 * (1 - R_1) * best_array
                VEC = S2 / S1
                
                if random.random() > Q:
                    # Phương trình 32 phần đầu
                    random_sol = random.choice(population)
                    random_array = np.array([random_sol[param] for param in self.param_ranges])
                    new_pos = best_array + beta1 * (np.exp(beta2)) * (random_array - current_array)
                else:
                    # Phương trình 32 phần thứ hai
                    new_pos = beta1 * VEC - best_array
            else:
                # Phương trình 33
                r1 = random.randint(0, self.population_size-1)
                r1_sol = population[r1]
                r1_array = np.array([r1_sol[param] for param in self.param_ranges])
                mbest_array = np.array([mbest[param] for param in self.param_ranges])
                current_array = np.array([current[param] for param in self.param_ranges])
                
                sign = 1 if random.random() > 0.5 else -1
                new_pos = (mbest_array * r1_array - sign * current_array) / (1 + (Beta * random.random()))
            
            # Chuyển đổi lại thành từ điển và cắt giá trị
            for j, param in enumerate(self.param_ranges):
                new_individual[param] = self._apply_bounds_and_type(param, new_pos[j])
            
            # Đánh giá và cập nhật (RMSE: giá trị thấp hơn là tốt hơn)
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val < fitness_values[i]:  # RMSE thấp hơn là tốt hơn
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness

    def optimize(self):
        """Thuật toán tối ưu hóa PUMA chính cho bài toán hồi quy"""
        # Khởi tạo quần thể
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Tốt nhất ban đầu (RMSE: giá trị thấp hơn là tốt hơn)
        best_idx = np.argmin(fitness_values)
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]
        initial_best_score = self.best_score
        self.best_scores_history.append(self.best_score)
        
        # Khởi tạo theo dõi kết quả lặp
        iteration_results = []
        
        # Tham số cho việc lựa chọn giai đoạn
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
        
        # Giai đoạn Chưa kinh nghiệm (3 lần lặp đầu tiên)
        for iteration in range(3):
            print(f"Iteration {iteration + 1}/3")
            
            # Khám phá
            pop_explore, fit_explore = self.exploration_phase(population, fitness_values)
            cost_explore = min(fit_explore)  # RMSE thấp hơn là tốt hơn
            
            # Khai thác
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            cost_exploit = min(fit_exploit)  # RMSE thấp hơn là tốt hơn
            
            # Kết hợp và chọn nghiệm tốt nhất
            population = population + pop_explore + pop_exploit
            fitness_values = fitness_values + fit_explore + fit_exploit
            indices = np.argsort(fitness_values)[:self.population_size]  # Sắp xếp tăng dần (RMSE thấp hơn là tốt hơn)
            population = [population[i] for i in indices]
            fitness_values = [fitness_values[i] for i in indices]
            
            # Cập nhật tốt nhất (RMSE thấp hơn là tốt hơn)
            if fitness_values[0] < self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
                print(f"New best RMSE: {self.best_score:.4f}")
                self.best_scores_history.append(self.best_score)
            
            # Lưu kết quả lặp
            iteration_results.append({
                'iteration': iteration + 1,
                'best_rmse': self.best_score,
                'best_params': self.best_individual.copy(),
                'population_mean_rmse': np.mean(fitness_values),
                'population_min_rmse': np.min(fitness_values),
                'population_max_rmse': np.max(fitness_values),
                'phase': 'Unexperienced'
            })
        
        # Khởi tạo chi phí chuỗi
        seq_cost_explore[0] = max(0.01, abs(initial_best_score - cost_explore))
        seq_cost_exploit[0] = max(0.01, abs(initial_best_score - cost_exploit))
        
        # Thêm chi phí khác 0 vào PF_F3
        for cost in seq_cost_explore + seq_cost_exploit:
            if cost > 0.01:
                pf_f3.append(cost)
        
        # Tính toán điểm số ban đầu
        f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
        f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
        f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
        f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
        score_explore = pf[0] * f1_explore + pf[1] * f2_explore
        score_exploit = pf[0] * f1_exploit + pf[1] * f2_exploit
        
        # Giai đoạn Có kinh nghiệm
        for iteration in range(3, self.generations):
            print(f"Iteration {iteration + 1}/{self.generations}")
            
            if score_explore > score_exploit:
                # Khám phá
                population, fitness_values = self.exploration_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[1] += 1
                unselected[0] = 1
                f3_explore = pf[2]
                f3_exploit += pf[2]
                
                # Cập nhật chi phí chuỗi (RMSE thấp hơn là tốt hơn)
                if fitness_values[0] < self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_explore = [max(0.01, cost_diff)] + seq_cost_explore[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            else:
                # Khai thác
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[0] += 1
                unselected[1] = 1
                f3_explore += pf[2]
                f3_exploit = pf[2]
                
                # Cập nhật chi phí chuỗi (RMSE thấp hơn là tốt hơn)
                if fitness_values[0] < self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_exploit = [max(0.01, cost_diff)] + seq_cost_exploit[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            
            # Cập nhật nghiệm tốt nhất (RMSE thấp hơn là tốt hơn)
            if fitness_values[0] < self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
                print(f"New best RMSE: {self.best_score:.4f}")
                self.best_scores_history.append(self.best_score)
            
            # Cập nhật chuỗi thời gian nếu giai đoạn thay đổi
            if flag_change != (1 if score_explore > score_exploit else 2):
                flag_change = 1 if score_explore > score_exploit else 2
                seq_time_explore = [count_select[0]] + seq_time_explore[:2]
                seq_time_exploit = [count_select[1]] + seq_time_exploit[:2]
            
            # Cập nhật điểm số
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
            
            # Lưu kết quả lặp
            iteration_results.append({
                'iteration': iteration + 1,
                'best_rmse': self.best_score,
                'best_params': self.best_individual.copy(),
                'population_mean_rmse': np.mean(fitness_values),
                'population_min_rmse': np.min(fitness_values),
                'population_max_rmse': np.max(fitness_values),
                'phase': 'Exploration' if score_explore > score_exploit else 'Exploitation'
            })
        
        return self.best_individual, self.best_score

def main():
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Read {len(df)} rows of data")
        
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        label_column = 'label_column'  # Biến mục tiêu cho bài toán hồi quy
        
        # Kiểm tra cột bị thiếu
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            return
        
        # Chuẩn bị dữ liệu
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Xử lý giá trị bị thiếu
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Khởi tạo và chạy bộ tối ưu hóa PUMA cho Hồi quy XGBoost
        print("Starting PUMA optimization for regression...")
        optimizer = PUMAOptimizer(X, y, population_size=15, generations=10)
        best_params, best_score = optimizer.optimize()
        
        print("\nOptimization completed!")
        print(f"Best RMSE score: {best_score:.4f}")
        print("\nBest parameters:")
        for param, value in best_params.items():
            if param != '_metrics':  # Bỏ qua khóa metrics
                print(f"  {param}: {value}")
            
        # Xuất dữ liệu hội tụ ra CSV
        convergence_data = pd.DataFrame({
            'Iteration': range(1, len(optimizer.best_scores_history) + 1),
            'Best_RMSE_Score': optimizer.best_scores_history
        })
        convergence_data.to_csv('po_xgb_convergence.csv', index=False)
        print("\nConvergence data exported to 'po_xgb_convergence.csv'")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
