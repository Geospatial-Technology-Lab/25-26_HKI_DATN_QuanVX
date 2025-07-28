import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
import joblib


class PSOXGBoostOptimizer:
    """Tối ưu hóa Bầy đàn Hạt (PSO) cho việc điều chỉnh siêu tham số XGBoost."""
    
    def __init__(self, X, y, n_particles=30, n_iterations=50):
        """Khởi tạo bộ tối ưu hóa PSO."""
        self.X = X
        self.y = y
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # Chuẩn bị dữ liệu
        self._prepare_data()
        
        # Tham số PSO
        self.w = 0.9    # Trọng số quán tính
        self.c1 = 2.0   # Tham số nhận thức
        self.c2 = 2.0   # Tham số xã hội
        self.w_min = 0.4 # Trọng số quán tính tối thiểu
        
        # Không gian tìm kiếm tham số - Bộ tham số mở rộng cho XGBoost
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
        
        # Khởi tạo bầy đàn
        self._initialize_swarm()
        
        # Kết quả tối ưu hóa
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.optimization_history = []
        self.avg_scores_history = []
    
    def _prepare_data(self):
        """Chuẩn bị và chia dữ liệu cho huấn luyện."""
        # Xử lý giá trị thiếu
        if np.isnan(self.X).any():
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Chia dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2,
            stratify=self.y
        )
        
        # Chuẩn hóa đặc trưng
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def _initialize_swarm(self):
        """Khởi tạo vị trí và vận tốc của các hạt."""
        # Khởi tạo vị trí
        self.positions = []
        for _ in range(self.n_particles):
            position = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'int':
                    position[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
                elif range_info['type'] == 'float':
                    position[param] = np.random.uniform(range_info['min'], range_info['max'])
                elif range_info['type'] == 'log_uniform':
                    log_min = np.log10(range_info['min'])
                    log_max = np.log10(range_info['max'])
                    position[param] = 10 ** np.random.uniform(log_min, log_max)
                elif range_info['type'] == 'choice':
                    position[param] = np.random.choice(range_info['options'])
                elif range_info['type'] == 'tuple_choice':
                    position[param] = np.random.choice(range_info['options'])
            self.positions.append(position)
        
        # Khởi tạo vận tốc
        self.velocities = []
        for _ in range(self.n_particles):
            velocity = {}
            for param, range_info in self.param_ranges.items():
                max_velocity = (range_info['max'] - range_info['min']) * 0.1
                velocity[param] = np.random.uniform(-max_velocity, max_velocity)
            self.velocities.append(velocity)
        
        # Khởi tạo tốt nhất cá nhân
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, -np.inf)
    
    def _evaluate_fitness(self, position):
        """Đánh giá độ phù hợp của hạt bằng cách sử dụng kiểm định chéo."""
        try:
            xgb = XGBClassifier(
                n_estimators=int(position['n_estimators']),
                max_depth=int(position['max_depth']),
                learning_rate=position['learning_rate'],
                subsample=position['subsample'],
                colsample_bytree=position['colsample_bytree'],
                min_child_weight=int(position['min_child_weight']),
                gamma=position['gamma'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            cv_scores = cross_val_score(
                xgb, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1', n_jobs=-1
            )
            
            return float(np.mean(cv_scores))
        except:
            return -np.inf
    
    def _update_particle(self, particle_idx):
        """Cập nhật vận tốc và vị trí của hạt."""
        # Cập nhật vận tốc
        w = self.w - (self.w - self.w_min) * (particle_idx / self.n_particles)
        
        for param, range_info in self.param_ranges.items():
            # Công thức cập nhật vận tốc PSO chuẩn
            r1, r2 = np.random.random(2)
            cognitive = self.c1 * r1 * (self.personal_best_positions[particle_idx][param] - 
                                      self.positions[particle_idx][param])
            social = self.c2 * r2 * (self.global_best_position[param] - 
                                   self.positions[particle_idx][param])
            
            self.velocities[particle_idx][param] = (w * self.velocities[particle_idx][param] + 
                                                  cognitive + social)
            
            # Cập nhật vị trí
            self.positions[particle_idx][param] += self.velocities[particle_idx][param]
            
            # Giới hạn vị trí trong phạm vi
            self.positions[particle_idx][param] = np.clip(
                self.positions[particle_idx][param],
                range_info['min'],
                range_info['max']
            )
            
            # Làm tròn tham số nguyên
            if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                self.positions[particle_idx][param] = int(self.positions[particle_idx][param])
    
    def optimize(self):
        """Thực thi thuật toán tối ưu hóa PSO."""
        print("Bắt đầu tối ưu hóa PSO...")
        print(f"Tập dữ liệu: {len(self.X)} mẫu, {self.X.shape[1]} đặc trưng")
        print(f"Phân bố lớp: {np.bincount(self.y)}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Đánh giá bầy đàn ban đầu
        for i in range(self.n_particles):
            score = self._evaluate_fitness(self.positions[i])
            self.personal_best_scores[i] = score
            
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()
        
        # Vòng lặp tối ưu hóa chính
        for iteration in range(self.n_iterations):
            # Cập nhật các hạt
            current_scores = []
            for i in range(self.n_particles):
                self._update_particle(i)
                
                # Đánh giá vị trí mới
                score = self._evaluate_fitness(self.positions[i])
                current_scores.append(score)
                
                # Cập nhật tốt nhất cá nhân
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                    # Cập nhật tốt nhất toàn cục
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[i].copy()
            
            # Tính điểm trung bình
            avg_score = np.mean(current_scores)
            self.avg_scores_history.append(avg_score)
            
            # Ghi log tiến trình
            print(f"Lần lặp {iteration + 1:2d}/{self.n_iterations}: "
                  f"F1 tốt nhất={self.global_best_score:.4f}, F1 trung bình={avg_score:.4f}")
            
            # Lưu lịch sử
            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_score': self.global_best_score,
                'best_params': self.global_best_position.copy(),
                'population_mean_score': np.mean(current_scores),
                'population_min_score': np.min(current_scores),
                'population_max_score': np.max(current_scores),
                'inertia_weight': self.w,
                'cognitive_param': self.c1,
                'social_param': self.c2
            })
        
        optimization_time = time.time() - start_time
        
        print("-" * 60)
        print(f"Tối ưu hóa hoàn thành trong {optimization_time:.2f} giây")
        print(f"Điểm F1 tốt nhất: {self.global_best_score:.4f}")
        print("Tham số tối ưu:")
        for param, value in self.global_best_position.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        
        # Xuất dữ liệu hội tụ ra CSV
        convergence_data = pd.DataFrame(self.optimization_history)
        convergence_data.to_csv('pso_xgb_iterations.csv', index=False)
        print("\nDữ liệu hội tụ đã được xuất ra 'pso_xgb_iterations.csv'")
        
        return self.global_best_position, self.global_best_score
    
    def evaluate_test_performance(self):
        """Huấn luyện mô hình cuối cùng và đánh giá trên tập kiểm tra."""
        if not self.global_best_position:
            raise ValueError("Không có kết quả tối ưu hóa nào. Hãy chạy optimize() trước.")
        
        # Huấn luyện mô hình cuối cùng
        final_model = XGBClassifier(
            n_estimators=int(self.global_best_position['n_estimators']),
            max_depth=int(self.global_best_position['max_depth']),
            learning_rate=self.global_best_position['learning_rate'],
            subsample=self.global_best_position['subsample'],
            colsample_bytree=self.global_best_position['colsample_bytree'],
            min_child_weight=int(self.global_best_position['min_child_weight']),
            gamma=self.global_best_position['gamma'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        final_model.fit(self.X_train_scaled, self.y_train)
        
        # Đánh giá trên tập kiểm tra
        y_pred = final_model.predict(self.X_test_scaled)
        y_prob = final_model.predict_proba(self.X_test_scaled)[:, 1]
        
        test_metrics = {
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_prob),
            'accuracy': accuracy_score(self.y_test, y_pred),
            'model': final_model,
            'best_params': self.global_best_position
        }
        
        print("\nHiệu suất trên Tập Kiểm tra:")
        print(f"Điểm F1:   {test_metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"Độ chính xác: {test_metrics['accuracy']:.4f}")
        
        return test_metrics
    
    def plot_optimization_progress(self):
        """Vẽ biểu đồ tiến trình tối ưu hóa."""
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.optimization_history) + 1)
        best_scores = [h['best_score'] for h in self.optimization_history]
        
        plt.plot(iterations, best_scores, 'b-', label='Điểm F1 Tốt nhất')
        plt.plot(iterations, self.avg_scores_history, 'r--', label='Điểm F1 Trung bình')
        
        plt.title('Tiến trình Tối ưu hóa PSO')
        plt.xlabel('Lần lặp')
        plt.ylabel('Điểm F1')
        plt.grid(True)
        plt.legend()
        plt.show()


def load_and_preprocess_data(file_path):
    """Tải và tiền xử lý dữ liệu từ file CSV."""
    try:
        # Đọc CSV với dấu phân cách chấm phẩy
        df = pd.read_csv(file_path, sep=';', na_values='<Null>')
        print(f"Đã tải tập dữ liệu với {len(df)} hàng và {len(df.columns)} cột")
        
        # Cột đặc trưng cho dự đoán lũ lụt
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Chuyển đổi Yes/No thành 1/0
        df[label_column] = (df[label_column] == 'Yes').astype(int)
        
        # Thay thế dấu phẩy bằng dấu chấm trong các cột số và chuyển đổi thành float
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        X = df[feature_columns].values
        y = df[label_column].values
        
        return X, y, feature_columns
        
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"LỖI: {str(e)}")
        return None, None, None


def main():
    """Hàm thực thi chính."""
    # Tải dữ liệu
    X, y, feature_names = load_and_preprocess_data("training.csv")
    if X is None:
        return
    
    # Khởi tạo và chạy bộ tối ưu hóa
    optimizer = PSOXGBoostOptimizer(
        X=X, 
        y=y, 
        n_particles=30, 
        n_iterations=50
    )
    
    # Tối ưu hóa siêu tham số
    best_params, best_score = optimizer.optimize()
    
    # Vẽ biểu đồ tiến trình tối ưu hóa
    optimizer.plot_optimization_progress()
    
    # Đánh giá mô hình cuối cùng
    test_results = optimizer.evaluate_test_performance()
    
    # Lưu mô hình tốt nhất
    joblib.dump(test_results['model'], 'pso_xgb_model.joblib')
    print("\nMô hình đã được lưu thành: pso_xgb_model.joblib")


if __name__ == "__main__":
    main()