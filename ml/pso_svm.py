import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time


class PSOSVMOptimizer:
    """Tối ưu hóa Swarm Particles cho việc điều chỉnh siêu tham số SVM."""
    
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
        
        # Không gian tìm kiếm tham số - Bộ tham số mở rộng cho SVM
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
        
        # Khởi tạo bầy đàn
        self._initialize_swarm()
        
        # Kết quả tối ưu hóa
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.best_scores_history = []
    
    def _prepare_data(self):
        """Chuẩn bị và chia dữ liệu để huấn luyện."""
        # Xử lý giá trị bị thiếu
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
        print(f"Sử dụng tất cả {self.X_train_scaled.shape[1]} đặc trưng")
    
    def _initialize_swarm(self):
        """Khởi tạo vị trí và vận tốc của các hạt."""
        # Khởi tạo vị trí
        self.positions = []
        for _ in range(self.n_particles):
            position = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'continuous':
                    if range_info.get('log_scale', False):
                        position[param] = np.random.uniform(
                            np.log10(range_info['min']), 
                            np.log10(range_info['max'])
                        )
                    else:
                        position[param] = np.random.uniform(
                            range_info['min'], 
                            range_info['max']
                        )
                elif range_info['type'] == 'integer':
                    position[param] = np.random.randint(
                        range_info['min'], 
                        range_info['max'] + 1
                    )
                else:  # discrete
                    position[param] = np.random.choice(range_info['values'])
            self.positions.append(position)
        
        # Khởi tạo vận tốc
        self.velocities = []
        for _ in range(self.n_particles):
            velocity = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'continuous':
                    if range_info.get('log_scale', False):
                        max_velocity = (np.log10(range_info['max']) - np.log10(range_info['min'])) * 0.1
                    else:
                        max_velocity = (range_info['max'] - range_info['min']) * 0.1
                    velocity[param] = np.random.uniform(-max_velocity, max_velocity)
                elif range_info['type'] == 'integer':
                    max_velocity = (range_info['max'] - range_info['min']) * 0.1
                    velocity[param] = np.random.uniform(-max_velocity, max_velocity)
                else:  # discrete
                    velocity[param] = 0
            self.velocities.append(velocity)
        
        # Khởi tạo tốt nhất cá nhân
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, -np.inf)
    
    def _evaluate_fitness(self, position):
        """Đánh giá độ phù hợp của hạt."""
        try:
            # Chuyển đổi tham số log-scale
            params = position.copy()
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'continuous' and range_info.get('log_scale', False):
                    params[param] = 10 ** params[param]
            
            svm = SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'] if params['kernel'] != 'linear' else 'auto',
                degree=params['degree'] if params['kernel'] == 'poly' else 3,
                coef0=params['coef0'] if params['kernel'] in ['poly', 'sigmoid'] else 0.0,
                probability=True
            )
            
            cv_scores = cross_val_score(
                svm, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1'
            )
            
            return float(np.mean(cv_scores))
        except:
            return -np.inf
    
    def _update_particle(self, particle_idx):
        """Cập nhật vận tốc và vị trí của hạt."""
        # Cập nhật vận tốc
        w = self.w - (self.w - self.w_min) * (particle_idx / self.n_particles)
        
        for param, range_info in self.param_ranges.items():
            if range_info['type'] != 'discrete':
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
                if range_info['type'] == 'continuous':
                    if range_info.get('log_scale', False):
                        self.positions[particle_idx][param] = np.clip(
                            self.positions[particle_idx][param],
                            np.log10(range_info['min']),
                            np.log10(range_info['max'])
                        )
                    else:
                        self.positions[particle_idx][param] = np.clip(
                            self.positions[particle_idx][param],
                            range_info['min'],
                            range_info['max']
                        )
                else:  # integer
                    self.positions[particle_idx][param] = int(np.clip(
                        self.positions[particle_idx][param],
                        range_info['min'],
                        range_info['max']
                    ))
            else:  # discrete parameters
                if np.random.random() < 0.1:  # 10% khả năng thay đổi
                    self.positions[particle_idx][param] = np.random.choice(range_info['values'])
    
    def optimize(self):
        """Thực thi thuật toán tối ưu hóa PSO."""
        print("Bắt đầu tối ưu hóa PSO...")
        print(f"Tập dữ liệu: {len(self.X)} mẫu, {self.X.shape[1]} đặc trưng")
        print(f"Phân bố lớp: {np.bincount(self.y)}")
        print(f"Số particles: {self.n_particles}, Số iterations: {self.n_iterations}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Đánh giá bầy đàn ban đầu - đơn giản hóa
        print("Đánh giá bầy đàn ban đầu...")
        initial_scores = []
        for position in self.positions:
            score = self._evaluate_fitness(position)
            initial_scores.append(score)
        
        for i, score in enumerate(initial_scores):
            self.personal_best_scores[i] = score
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()
        
        print(f"Điểm F1 tốt nhất ban đầu: {self.global_best_score:.4f}")
        
        # Vòng lặp tối ưu hóa chính - đơn giản hóa
        for iteration in range(self.n_iterations):
            
            # Cập nhật các hạt
            for i in range(self.n_particles):
                self._update_particle(i)
            
            # Đánh giá vị trí mới - đơn giản hóa
            current_scores = []
            for position in self.positions:
                score = self._evaluate_fitness(position)
                current_scores.append(score)
            
            # Cập nhật best scores
            for i, score in enumerate(current_scores):
                # Cập nhật tốt nhất cá nhân
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                    # Cập nhật tốt nhất toàn cục
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[i].copy()
            
            # Lưu điểm tốt nhất
            self.best_scores_history.append(self.global_best_score)
            
            # Progress tracking
            print(f"Lặp {iteration + 1:2d}/{self.n_iterations}: "
                  f"F1 tốt nhất={self.global_best_score:.4f}")
        
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
        
        return self.global_best_position, self.global_best_score
    
    def evaluate_test_performance(self):
        """Huấn luyện mô hình cuối cùng và đánh giá trên tập kiểm tra."""
        if not self.global_best_position:
            raise ValueError("Không có kết quả tối ưu hóa. Hãy chạy optimize() trước.")
        
        # Chuyển đổi tham số
        params = self.global_best_position.copy()
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'continuous' and range_info.get('log_scale', False):
                params[param] = 10 ** params[param]
        
        # Huấn luyện mô hình cuối cùng
        final_model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=params['gamma'] if params['kernel'] != 'linear' else 'auto',
            degree=params['degree'] if params['kernel'] == 'poly' else 3,
            coef0=params['coef0'] if params['kernel'] in ['poly', 'sigmoid'] else 0.0,
            probability=True
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
            'best_params': params
        }
        
        print("\nHiệu suất trên tập kiểm tra:")
        print(f"Điểm F1: {test_metrics['f1_score']:.4f}")
        print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Độ chính xác: {test_metrics['accuracy']:.4f}")
        
        return test_metrics
    
    def plot_optimization_progress(self):
        """Vẽ biểu đồ tiến trình tối ưu hóa."""
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.best_scores_history) + 1)
        
        plt.plot(iterations, self.best_scores_history, 'b-', label='Điểm F1 Tốt nhất')
        
        plt.title('Tiến trình Tối ưu hóa PSO')
        plt.xlabel('Lần lặp')
        plt.ylabel('Điểm F1')
        plt.grid(True)
        plt.legend()
        plt.show()


def load_and_preprocess_data(file_path):
    """Tải và tiền xử lý dữ liệu từ file CSV."""
    try:
        # Đọc CSV với dấu phân cách dấu chấm phẩy
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
        
        # Thay thế dấu phẩy bằng dấu chấm trong cột số và chuyển đổi thành float
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
    print("Khởi động PSO-SVM Optimizer")
    print("=" * 60)
    
    # Tải dữ liệu
    X, y, _ = load_and_preprocess_data("training.csv")
    if X is None:
        return
    
    # Khởi tạo và chạy bộ tối ưu hóa
    optimizer = PSOSVMOptimizer(X=X, y=y, n_particles=30, n_iterations=50)
    
    try:
        # Tối ưu hóa siêu tham số
        best_params, best_score = optimizer.optimize()
        
        # Vẽ biểu đồ tiến trình tối ưu hóa
        print("\nĐang vẽ biểu đồ tiến trình...")
        optimizer.plot_optimization_progress()
        
        # Đánh giá mô hình cuối cùng
        print("\nĐánh giá mô hình cuối cùng...")
        test_results = optimizer.evaluate_test_performance()
        
        # Tóm tắt kết quả
        print("\n" + "="*60)
        print("TÓM TẮT KẾT QUẢ:")
        print(f"Điểm F1 tốt nhất: {best_score:.4f}")
        print(f"ROC AUC: {test_results['roc_auc']:.4f}")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTối ưu hóa bị dừng bởi người dùng")
    except Exception as e:
        print(f"\nLỗi trong quá trình tối ưu hóa: {e}")


if __name__ == "__main__":
    main()