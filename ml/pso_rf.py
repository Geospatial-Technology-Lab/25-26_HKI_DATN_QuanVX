import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
import joblib

class PSORandomForestOptimizer:
    """Tối ưu hóa Bầy đàn Hạt (PSO) cho việc điều chỉnh siêu tham số Random Forest."""
    
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
        
        # Không gian tìm kiếm tham số - Bộ tham số mở rộng cho Random Forest
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
            'max_depth': {'type': 'int', 'min': 3, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
            'max_features': {'type': 'choice', 'options': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9]},
            'bootstrap': {'type': 'choice', 'options': [True, False]},
            'max_leaf_nodes': {'type': 'int', 'min': 10, 'max': 1000},
            'min_impurity_decrease': {'type': 'float', 'min': 0.0, 'max': 0.2},
            'class_weight': {'type': 'choice', 'options': [None, 'balanced', 'balanced_subsample']},
            'criterion': {'type': 'choice', 'options': ['gini', 'entropy']}
        }
        
        # Khởi tạo bầy đàn
        self._initialize_swarm()
        
        # Kết quả tối ưu hóa
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.optimization_history = []
        self.avg_scores_history = []
    
    def _prepare_data(self):
        """Chuẩn bị và chia dữ liệu để huấn luyện."""
        # Xử lý giá trị thiếu
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.values
        if isinstance(self.y, pd.Series):
            self.y = self.y.values
            
        # Xử lý NaN
        imputer = SimpleImputer(strategy='median')
        self.X = imputer.fit_transform(self.X)
        
        # Chia dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Kích thước dữ liệu huấn luyện: {self.X_train_scaled.shape}")
        print(f"Kích thước dữ liệu kiểm tra: {self.X_test_scaled.shape}")
    
    def _initialize_swarm(self):
        """Khởi tạo bầy đàn hạt."""
        self.particles = []
        
        for _ in range(self.n_particles):
            particle = {
                'position': self._generate_random_params(),
                'velocity': {},
                'best_position': {},
                'best_score': -np.inf
            }
            
            # Khởi tạo vận tốc
            for param in self.param_ranges:
                particle['velocity'][param] = 0.0
            
            self.particles.append(particle)
    
    def _generate_random_params(self):
        """Tạo bộ tham số ngẫu nhiên."""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
            elif range_info['type'] == 'float':
                params[param] = np.random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                params[param] = 10 ** np.random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                params[param] = np.random.choice(range_info['options'])
            elif range_info['type'] == 'tuple_choice':
                params[param] = np.random.choice(range_info['options'])
        
        return params
    
    def _evaluate_particle(self, params):
        """Đánh giá một hạt (bộ tham số)."""
        try:
            # Tạo dictionary chỉ chứa các tham số hợp lệ cho RandomForestClassifier
            model_params = {}
            valid_params = [
                'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                'max_features', 'bootstrap', 'max_leaf_nodes', 'min_impurity_decrease',
                'class_weight', 'criterion'
            ]
            
            for param in valid_params:
                if param in params:
                    model_params[param] = params[param]
            
            rf = RandomForestClassifier(
                **model_params,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                rf, self.X_train_scaled, self.y_train,
                cv=3, scoring='f1', n_jobs=-1
            )
            
            return np.mean(cv_scores)
            
        except Exception as e:
            print(f"Lỗi khi đánh giá hạt: {str(e)}")
            return -np.inf
    
    def optimize(self):
        """Thực hiện tối ưu hóa PSO."""
        print(f"Bắt đầu tối ưu hóa PSO với {self.n_particles} hạt và {self.n_iterations} vòng lặp...")
        
        for iteration in range(self.n_iterations):
            print(f"\nVòng lặp {iteration + 1}/{self.n_iterations}")
            
            # Đánh giá tất cả các hạt
            scores = []
            for i, particle in enumerate(self.particles):
                score = self._evaluate_particle(particle['position'])
                scores.append(score)
                
                # Cập nhật best cá nhân
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # Cập nhật best toàn cục
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle['position'].copy()
                    print(f"*** Tìm thấy giải pháp tốt hơn! Điểm số: {score:.4f} ***")
            
            # Lưu lịch sử
            self.optimization_history.append(self.global_best_score)
            self.avg_scores_history.append(np.mean(scores))
            
            print(f"Điểm số tốt nhất: {self.global_best_score:.4f}")
            print(f"Điểm số trung bình: {np.mean(scores):.4f}")
            
            # Cập nhật vị trí và vận tốc
            self._update_particles()
            
            # Giảm trọng số quán tính
            self.w = max(self.w_min, self.w - (self.w - self.w_min) / self.n_iterations)
        
        print(f"\nTối ưu hóa hoàn thành!")
        print(f"Điểm số tốt nhất: {self.global_best_score:.4f}")
        
        return self.global_best_position, self.global_best_score
    
    def _update_particles(self):
        """Cập nhật vị trí và vận tốc của các hạt."""
        for particle in self.particles:
            for param in self.param_ranges:
                # Chỉ cập nhật tham số số
                if self.param_ranges[param]['type'] in ['int', 'float', 'log_uniform']:
                    # Cập nhật vận tốc
                    r1, r2 = np.random.random(), np.random.random()
                    
                    cognitive = self.c1 * r1 * (particle['best_position'][param] - particle['position'][param])
                    social = self.c2 * r2 * (self.global_best_position[param] - particle['position'][param])
                    
                    particle['velocity'][param] = (self.w * particle['velocity'][param] + 
                                                 cognitive + social)
                    
                    # Cập nhật vị trí
                    particle['position'][param] += particle['velocity'][param]
                    
                    # Ràng buộc trong phạm vi
                    param_range = self.param_ranges[param]
                    if param_range['type'] == 'log_uniform':
                        particle['position'][param] = np.clip(particle['position'][param], 
                                                            param_range['min'], param_range['max'])
                    else:
                        particle['position'][param] = np.clip(particle['position'][param], 
                                                            param_range['min'], param_range['max'])
                    
                    # Làm tròn cho tham số integer
                    if param_range['type'] == 'int':
                        particle['position'][param] = int(round(particle['position'][param]))
                        
                else:
                    # Đối với tham số categorical, chọn ngẫu nhiên thỉnh thoảng
                    if np.random.random() < 0.1:  # 10% cơ hội thay đổi
                        particle['position'][param] = self._generate_random_params()[param]
    
    def evaluate_final_model(self):
        """Đánh giá mô hình cuối cùng trên tập kiểm tra."""
        if not self.global_best_position:
            print("Không có mô hình tối ưu!")
            return None
        
        # Tạo dictionary chỉ chứa các tham số hợp lệ cho RandomForestClassifier
        model_params = {}
        valid_params = [
            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
            'max_features', 'bootstrap', 'max_leaf_nodes', 'min_impurity_decrease',
            'class_weight', 'criterion'
        ]
        
        for param in valid_params:
            if param in self.global_best_position:
                model_params[param] = self.global_best_position[param]
        
        # Huấn luyện mô hình tối ưu
        best_rf = RandomForestClassifier(
            **model_params,
            random_state=42,
            n_jobs=-1
        )
        
        best_rf.fit(self.X_train_scaled, self.y_train)
        
        # Dự đoán và đánh giá
        y_pred = best_rf.predict(self.X_test_scaled)
        y_prob = best_rf.predict_proba(self.X_test_scaled)[:, 1]
        
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print(f"\nKết quả trên tập kiểm tra:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Độ chính xác: {test_acc:.4f}")
        
        return {
            'model': best_rf,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.global_best_position
        }

def main():
    """Hàm chính"""
    print("Đọc dữ liệu từ file Excel...")
    
    # Thay đổi đường dẫn này thành file dữ liệu của bạn
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Đã đọc {len(df)} hàng dữ liệu")
        
        # Cột đặc trưng
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Cột nhãn
        label_column = 'label_column'  # 1 = lụt, 0 = không lụt
        
        # Kiểm tra cột thiếu
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            print(f"CẢNH BÁO: Các cột sau không tìm thấy: {missing_cols}")
            print(f"Các cột có sẵn: {list(df.columns)}")
            return
        
        # Chuẩn bị dữ liệu
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Khởi tạo và chạy tối ưu hóa PSO
        optimizer = PSORandomForestOptimizer(X, y, n_particles=20, n_iterations=30)
        
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nThời gian tối ưu hóa: {end_time - start_time:.2f} giây")
        
        if best_params:
            # Đánh giá mô hình cuối cùng
            print("\nĐánh giá mô hình cuối cùng trên tập kiểm tra:")
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print(f"\nKết quả cuối cùng:")
                print(f"F1-Score: {final_results['test_f1']:.4f}")
                print(f"AUC: {final_results['test_auc']:.4f}")
                print(f"Độ chính xác: {final_results['test_accuracy']:.4f}")
        else:
            print("\nTối ưu hóa thất bại.")
    
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        print("Vui lòng đảm bảo file Excel của bạn tồn tại tại đường dẫn đã chỉ định")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
