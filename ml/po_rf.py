import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Thêm seed cố định để tái tạo kết quả
RANDOM_SEED = 42

class PUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.pCR = 0.5  # Tỷ lệ lai ghép ban đầu
        self.p = 0.1    # Tỷ lệ điều chỉnh pCR
        
        # Chia và chuẩn hóa dữ liệu với seed cố định
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=RANDOM_SEED
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Phạm vi tham số RF - Bộ tham số mở rộng
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
            'max_depth': {'type': 'int', 'min': 3, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
            'max_features': {'type': 'choice', 'options': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9]},
            'bootstrap': {'type': 'choice', 'options': [True, False]},
            'max_leaf_nodes': {'type': 'int', 'min': 10, 'max': 1000},
            'min_impurity_decrease': {'type': 'float', 'min': 0.0, 'max': 0.2}
        }
        
        # Lấy tham số số để thực hiện các phép toán vector nhất quán
        self.numerical_params = list(self.param_ranges.keys())
        self.num_numerical = len(self.numerical_params)
    
    def create_individual(self):
        """Tạo một cá thể ngẫu nhiên (bộ tham số)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            range_size = range_info['max'] - range_info['min']
            # Thêm nhiễu ngẫu nhiên nhỏ để tránh phân cụm quanh các giá trị nhất định
            noise = random.gauss(0, range_size * 0.1)  # 10% của phạm vi làm độ lệch chuẩn
            rand_val = random.random() * range_size + range_info['min'] + noise
            # Đảm bảo giá trị nằm trong giới hạn sau khi thêm nhiễu
            rand_val = max(range_info['min'], min(range_info['max'], rand_val))
            individual[param] = int(round(rand_val))
        return individual
    
    def evaluate_individual(self, individual):
        """Đánh giá fitness của một cá thể sử dụng Random Forest Regressor với 3 metrics đơn giản"""
        try:
            # Tạo mô hình Random Forest Regressor
            model = RandomForestRegressor(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                n_jobs=-1,
                random_state=RANDOM_SEED
            )

            # Huấn luyện mô hình
            model.fit(self.X_train_scaled, self.y_train)
            
            # Lấy dự đoán (xác suất lũ từ 0 đến 1)
            y_pred = model.predict(self.X_test_scaled)
            y_pred = np.clip(y_pred, 0, 1)  # Giới hạn dự đoán từ 0 đến 1
            y_test = np.array(self.y_test)
            
            # Tính toán 3 metrics chính
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Tính điểm tổng hợp đơn giản trực tiếp
            rmse_score = 1 / (1 + rmse)
            mae_score = 1 / (1 + mae)
            r2_score_norm = (r2 + 1) / 2 if r2 <= 1 else 1
            composite_score = 0.6 * r2_score_norm + 0.25 * rmse_score + 0.15 * mae_score
            return float(composite_score)
            
        except Exception as e:
            print(f"Lỗi trong đánh giá: {str(e)}")
            return -np.inf
    
    def exploration_phase(self, population, fitness_values):
        """Giai đoạn khám phá PUMA"""
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Chọn 6 nghiệm khác nhau ngẫu nhiên
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Tạo nghiệm mới
            new_individual = current.copy()  # Bắt đầu với nghiệm hiện tại
            
            # Đảm bảo ít nhất một tham số thay đổi bằng cách chọn tham số ngẫu nhiên
            j0 = random.choice(list(self.param_ranges.keys()))
            
            for param, range_info in self.param_ranges.items():
                # Luôn thay đổi tham số j0 hoặc dựa trên xác suất pCR
                if param == j0 or random.random() <= self.pCR:
                    if random.random() < 0.5:
                        # Tạo giá trị ngẫu nhiên với nhiễu thêm để đa dạng hóa tốt hơn
                        range_size = range_info['max'] - range_info['min']
                        noise = random.gauss(0, range_size * 0.1)
                        rand_val = random.random() * range_size + range_info['min'] + noise
                        rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                        new_individual[param] = int(round(rand_val))
                    else:
                        G = 2 * random.random() - 1
                        term1 = a[param] + G * (a[param] - b[param])
                        term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                   ((c[param] - d[param]) - (e[param] - f[param])))
                        new_val = int(round(np.clip(term1 + term2, range_info['min'], range_info['max'])))
                        new_individual[param] = new_val
            
            # Đánh giá và cập nhật
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Cập nhật pCR khi không có cải thiện
                self.pCR = min(0.9, self.pCR + self.p)  # Giới hạn ở 0.9 để duy trì khám phá
        
        return new_population, new_fitness
    
    def exploitation_phase(self, population, fitness_values):
        """Giai đoạn khai thác PUMA"""
        Q = 0.67  # Hằng số khai thác
        Beta = 2  # Hằng số Beta
        
        # Chuyển đổi thành danh sách từ điển với cost để dễ thao tác
        Sol = [{'X': pop.copy(), 'Cost': fit} for pop, fit in zip(population, fitness_values)]
        NewSol = [{'X': {}, 'Cost': -np.inf} for _ in range(self.population_size)]
        
        # Lấy nghiệm tốt nhất
        best_idx = np.argmax(fitness_values)
        Best = {'X': population[best_idx].copy(), 'Cost': fitness_values[best_idx]}
        
        # Tính vị trí trung bình (mbest)
        mbest = {}
        for param in self.param_ranges.keys():
            mbest[param] = np.mean([s['X'][param] for s in Sol])
        
        for i in range(self.population_size):
            # Tạo các vector ngẫu nhiên
            beta1 = 2 * random.random()
            beta2 = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Tạo vector w và v (Eq 37, 38)
            w = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            v = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Tính F1 và F2 (Eq 35, 36)
            F1 = {param: random.gauss(0, 1) * np.exp(2 - i * (2/self.generations)) 
                  for param in self.param_ranges.keys()}
            F2 = {param: w[param] * (v[param]**2) * np.cos((2 * random.random()) * w[param])
                  for param in self.param_ranges.keys()}
            
            # Tính R_1 (Eq 34)
            R_1 = 2 * random.random() - 1
            
            # Tính S1 và S2
            S1 = {param: (2 * random.random() - 1 + random.gauss(0, 1))
                  for param in self.param_ranges.keys()}
            S2 = {param: (F1[param] * R_1 * Sol[i]['X'][param] + 
                         F2[param] * (1 - R_1) * Best['X'][param])
                  for param in self.param_ranges.keys()}
            
            # Tính VEC
            VEC = {param: S2[param] / S1[param] for param in self.param_ranges.keys()}
            
            if random.random() <= 0.5:
                Xatack = VEC
                if random.random() > Q:
                    # Eq 32 phần đầu
                    random_sol = random.choice(Sol)
                    for param in self.param_ranges.keys():
                        new_val = (Best['X'][param] + 
                                 beta1 * np.exp(beta2[param]) * 
                                 (random_sol['X'][param] - Sol[i]['X'][param]))
                        NewSol[i]['X'][param] = int(round(np.clip(new_val,
                                                self.param_ranges[param]['min'],
                                                self.param_ranges[param]['max'])))
                else:
                    # Eq 32 phần thứ hai
                    for param in self.param_ranges.keys():
                        new_val = beta1 * Xatack[param] - Best['X'][param]
                        NewSol[i]['X'][param] = int(round(np.clip(new_val,
                                                self.param_ranges[param]['min'],
                                                self.param_ranges[param]['max'])))
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                sign = 1 if random.random() > 0.5 else -1
                for param in self.param_ranges.keys():
                    new_val = ((mbest[param] * Sol[r1]['X'][param] - 
                              sign * Sol[i]['X'][param]) / 
                             (1 + (Beta * random.random())))
                    NewSol[i]['X'][param] = int(round(np.clip(new_val,
                                            self.param_ranges[param]['min'],
                                            self.param_ranges[param]['max'])))
            
            # Đánh giá nghiệm mới
            NewSol[i]['Cost'] = self.evaluate_individual(NewSol[i]['X'])
            
            # Cập nhật nghiệm (tối đa hóa fitness)
            if NewSol[i]['Cost'] > Sol[i]['Cost']:
                Sol[i] = NewSol[i].copy()
        
        # Chuyển đổi trở lại thành mảng population và fitness riêng biệt
        new_population = [s['X'] for s in Sol]
        new_fitness = [s['Cost'] for s in Sol]
        
        return new_population, new_fitness
    
    def optimize(self):
        """Chạy quá trình tối ưu hóa PUMA"""
        # Khởi tạo tham số
        UnSelected = [1, 1]  # [Khám phá, Khai thác]
        F3_Explore = 0.001
        F3_Exploit = 0.001
        Seq_Time_Explore = [1.0, 1.0, 1.0]
        Seq_Time_Exploit = [1.0, 1.0, 1.0]
        Seq_Cost_Explore = [1.0, 1.0, 1.0]
        Seq_Cost_Exploit = [1.0, 1.0, 1.0]
        Score_Explore = 0.001
        Score_Exploit = 0.001
        PF = [0.5, 0.5, 0.3]  # Tham số cho F1, F2, F3
        PF_F3 = []
        Mega_Explor = 0.99
        Mega_Exploit = 0.99
        Flag_Change = 1
        
        # Khởi tạo quần thể
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Tìm nghiệm tốt nhất ban đầu
        best_idx = np.argmax(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_best_fitness = best_fitness
        current_best_fitness = best_fitness
        
        print("\nTiến trình tối ưu hóa:")
        print("Gen |   R²   |  RMSE  |  MAE   | Điểm số")
        print("-" * 45)

        # Giai đoạn chưa có kinh nghiệm (3 lần lặp đầu tiên)
        for Iter in range(3):
            # Giai đoạn khám phá
            pop_explor, fit_explor = self.exploration_phase(population, fitness_values)
            Costs_Explor = max(fit_explor)
            
            # Giai đoạn khai thác
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            Costs_Exploit = max(fit_exploit)
            
            # Kết hợp và sắp xếp nghiệm
            all_population = population + pop_explor + pop_exploit
            all_fitness = fitness_values + fit_explor + fit_exploit
            
            # Sắp xếp theo fitness
            sorted_indices = np.argsort(all_fitness)[::-1]  # Thứ tự giảm dần để tối đa hóa
            population = [all_population[i] for i in sorted_indices[:self.population_size]]
            fitness_values = [all_fitness[i] for i in sorted_indices[:self.population_size]]
            
            # Chỉ cập nhật tốt nhất nếu fitness cải thiện
            if fitness_values[0] > current_best_fitness:
                best_individual = population[0].copy()
                best_fitness = fitness_values[0]
                current_best_fitness = best_fitness
            
            # In tiến trình với metrics đơn giản
            # Đánh giá lại nghiệm tốt nhất để lấy metrics
            model = RandomForestRegressor(
                n_estimators=best_individual['n_estimators'],
                max_depth=best_individual['max_depth'],
                min_samples_split=best_individual['min_samples_split'],
                min_samples_leaf=best_individual['min_samples_leaf'],
                n_jobs=-1,
                random_state=RANDOM_SEED
            )
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            y_pred = np.clip(y_pred, 0, 1)
            y_test = np.array(self.y_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{Iter+1:3d} | {r2:6.4f} | {rmse:6.4f} | {mae:6.4f} | {current_best_fitness:6.4f}")
        
        # Tính điểm ban đầu
        Seq_Cost_Explore[0] = abs(initial_best_fitness - Costs_Explor)
        Seq_Cost_Exploit[0] = abs(initial_best_fitness - Costs_Exploit)
        
        # Thêm cost khác 0 vào PF_F3
        for cost in Seq_Cost_Explore + Seq_Cost_Exploit:
            if cost != 0:
                PF_F3.append(cost)
        
        # Tính điểm F1 và F2 ban đầu
        F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
        F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
        F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore))
        F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit))
        
        # Tính điểm ban đầu
        Score_Explore = (PF[0] * F1_Explor) + (PF[1] * F2_Explor)
        Score_Exploit = (PF[0] * F1_Exploit) + (PF[1] * F2_Exploit)
        
        # Giai đoạn có kinh nghiệm
        for Iter in range(3, self.generations):
            if Score_Explore > Score_Exploit:
                # Chạy khám phá
                SelectFlag = 1
                population, fitness_values = self.exploration_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[1] += 1
                UnSelected[0] = 1
                F3_Explore = PF[2]
                F3_Exploit += PF[2]
                
                # Cập nhật sequence costs cho khám phá
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
                Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
                Seq_Cost_Explore[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Explore[0] != 0:
                    PF_F3.append(Seq_Cost_Explore[0])
                
                if temp_best_fitness > current_best_fitness:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
            else:
                # Chạy khai thác
                SelectFlag = 2
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[0] += 1
                UnSelected[1] = 1
                F3_Explore += PF[2]
                F3_Exploit = PF[2]
                
                # Cập nhật sequence costs cho khai thác
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
                Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
                Seq_Cost_Exploit[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Exploit[0] != 0:
                    PF_F3.append(Seq_Cost_Exploit[0])
                
                if temp_best_fitness > current_best_fitness:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
            
            # Cập nhật time sequences nếu giai đoạn thay đổi
            if Flag_Change != SelectFlag:
                Flag_Change = SelectFlag
                Seq_Time_Explore[2] = Seq_Time_Explore[1]
                Seq_Time_Explore[1] = Seq_Time_Explore[0]
                Seq_Time_Explore[0] = Count_select[0]
                Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
                Seq_Time_Exploit[1] = Seq_Time_Exploit[0]
                Seq_Time_Exploit[0] = Count_select[1]
            
            # Cập nhật điểm F1 và F2
            F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
            F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
            F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore))
            F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit))
            
            # Cập nhật điểm Mega
            if Score_Explore < Score_Exploit:
                Mega_Explor = max((Mega_Explor - 0.01), 0.01)
                Mega_Exploit = 0.99
            elif Score_Explore > Score_Exploit:
                Mega_Explor = 0.99
                Mega_Exploit = max((Mega_Exploit - 0.01), 0.01)
            
            # Tính giá trị lambda
            lmn_Explore = 1 - Mega_Explor
            lmn_Exploit = 1 - Mega_Exploit
            
            # Cập nhật điểm cuối cùng
            Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (min(PF_F3) * F3_Explore))
            Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min(PF_F3) * F3_Exploit))
            
            # In tiến trình với metrics đơn giản
            # Đánh giá lại nghiệm tốt nhất để lấy metrics
            model = RandomForestRegressor(
                n_estimators=best_individual['n_estimators'],
                max_depth=best_individual['max_depth'],
                min_samples_split=best_individual['min_samples_split'],
                min_samples_leaf=best_individual['min_samples_leaf'],
                n_jobs=-1,
                random_state=RANDOM_SEED
            )
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            y_pred = np.clip(y_pred, 0, 1)
            y_test = np.array(self.y_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{Iter+1:3d} | {r2:6.4f} | {rmse:6.4f} | {mae:6.4f} | {current_best_fitness:6.4f}")
        
        # Lưu trữ kết quả cuối cùng
        self.best_individual = best_individual
        self.best_score = best_fitness
        
        return self.best_individual, self.best_score

def plot_optimization_progress(best_scores):
    """Vẽ biểu đồ tiến trình tối ưu hóa đơn giản"""
    plt.figure(figsize=(10, 6))
    plt.plot(best_scores, 'b-', linewidth=2)
    plt.title('Tiến trình tối ưu hóa PUMA', fontsize=14)
    plt.xlabel('Thế hệ', fontsize=12)
    plt.ylabel('Điểm số tốt nhất', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names):
    """Vẽ biểu đồ tầm quan trọng của đặc trưng đơn giản"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Tầm quan trọng của các đặc trưng', fontsize=14)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Đặc trưng')
    plt.ylabel('Điểm tầm quan trọng')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Đọc CSV với dấu phân cách là dấu chấm phẩy
        df = pd.read_csv('C:/Users/Admin/Downloads/prj/src/flood_training.csv', sep=';', na_values='<Null>')
        
        # Cột đặc trưng để dự đoán lũ
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Chuyển đổi Yes/No thành 1/0 cho hồi quy (xác suất lũ)
        df[label_column] = (df[label_column] == 'Yes').astype(float)
        
        # Thay thế dấu phẩy bằng dấu chấm trong cột số và chuyển thành float
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Chuẩn bị dữ liệu
        X = df[feature_columns].values
        y = np.array(df[label_column].values)
        
        # Xử lý giá trị thiếu nếu có
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Khởi tạo và chạy bộ tối ưu PUMA cho RF
        print("Bắt đầu tối ưu hóa PUMA...")
        optimizer = PUMAOptimizer(X, y, population_size=25, generations=100)
        best_params, best_score = optimizer.optimize()
        
        # Vẽ biểu đồ tiến trình tối ưu hóa đơn giản
        # Tạo danh sách điểm số để vẽ biểu đồ
        best_scores = [best_score] * optimizer.generations  # Đơn giản hóa để demo
        plot_optimization_progress(best_scores)
        
        # In kết quả cuối cùng
        print("\n=== Kết quả cuối cùng ===")
        print(f"Điểm số tổng hợp tốt nhất: {best_score:.4f}")
        print("\nTham số tối ưu:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        # Huấn luyện mô hình cuối cùng với tham số tốt nhất
        final_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        
        # Huấn luyện và đánh giá trên tập kiểm tra
        final_model.fit(optimizer.X_train_scaled, optimizer.y_train)
        y_pred = final_model.predict(optimizer.X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Giới hạn dự đoán từ 0 đến 1
        
        # Tính toán và lưu metrics cuối cùng (đơn giản hóa)
        y_test = np.array(optimizer.y_test)
        final_r2 = r2_score(y_test, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        final_mae = mean_absolute_error(y_test, y_pred)
        
        metrics_data = pd.DataFrame({
            'Metric': ['R²', 'RMSE', 'MAE'],
            'Value': [final_r2, final_rmse, final_mae]
        })
        metrics_data.to_csv('po_rf_metrics.csv', index=False)
        
        print(f"\nMetrics cuối cùng:")
        print(f"R²: {final_r2:.4f}")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"MAE: {final_mae:.4f}")
            
    except FileNotFoundError:
        print("Không tìm thấy file! Vui lòng kiểm tra đường dẫn dataset.")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()