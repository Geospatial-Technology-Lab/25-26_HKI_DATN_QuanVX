import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time
import joblib

class RandomSearchOptimizer:
    def __init__(self, X, y, n_iter=10, feature_names=None):

        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iter = n_iter
        self.best_params = None
        self.best_score = -np.inf
        self.history = []
        self.feature_names = feature_names
        
        # Chia dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y
        )
        
        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Phân phối tham số Random Forest - Bộ tham số mở rộng
        self.param_distributions = {
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
    
    def sample_parameters(self):
        """Lấy mẫu tham số ngẫu nhiên"""
        params = {}
        for param_name, param_config in self.param_distributions.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'int':
                    params[param_name] = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'float':
                    params[param_name] = random.uniform(param_config['min'], param_config['max'])
                elif param_config['type'] == 'log_uniform':
                    log_min = np.log10(param_config['min'])
                    log_max = np.log10(param_config['max'])
                    params[param_name] = 10 ** random.uniform(log_min, log_max)
                elif param_config['type'] == 'choice':
                    params[param_name] = random.choice(param_config['options'])
                elif param_config['type'] == 'tuple_choice':
                    params[param_name] = random.choice(param_config['options'])
            else:  # Danh sách lựa chọn (backward compatibility)
                params[param_name] = random.choice(param_config)
        
        return params
    
    def evaluate_parameters(self, params):
        """Đánh giá bộ tham số"""
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
            
            # Tạo RandomForest với tham số được lấy mẫu
            rf = RandomForestClassifier(
                **model_params,
                random_state=42,
                n_jobs=-1  # Sử dụng tất cả lõi có sẵn
            )
            
            # Thực hiện kiểm định chéo
            cv_scores = cross_val_score(
                rf, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1', n_jobs=-1
            )
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Lỗi khi đánh giá tham số: {str(e)}")
            print("Tham số:", params)
            return -np.inf
    
    def optimize(self):
        print("Bắt đầu tối ưu hóa Random Search...")
        print(f"Dữ liệu: {len(self.X)} điểm, {self.X.shape[1]} đặc trưng")
        
        # Chuyển đổi y thành numpy array nếu chưa phải
        if not isinstance(self.y, np.ndarray):
            self.y = np.array(self.y)   # đảm bảo kiểu dữ liệu của y phù hợp cho các thao tác tính toán.
            
        unique_labels = np.unique(self.y)   # dùng để biết trong dữ liệu có những lớp nào.
        label_counts = np.bincount(self.y.astype(int))  # dùng để đếm lớp lụt/không lụt tránh mất cân bằng lớp.

        print("Phân bố lớp:")

        for iteration in range(self.n_iter):
            try:
                print(f"\nLần lặp {iteration + 1}/{self.n_iter}")
                
                # Lấy mẫu tham số ngẫu nhiên
                params = self.sample_parameters() #lấy mẫu tham số ngẫu nhiên từ phân phối đã định nghĩa.
                
                # Đánh giá tham số
                score = self.evaluate_parameters(params) # đánh giá bộ tham số bằng cách sử dụng cross-validation.
                
                # Cập nhật tốt nhất nếu tốt hơn
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                    print("\nTìm thấy giải pháp tốt nhất mới!")
                    print(f"Tham số:")
                    for param, value in params.items():
                        print(f"  {param}: {value}")
                
                print(f"\nĐiểm tốt nhất trong lần lặp {iteration + 1}: {self.best_score:.4f}")
                print(f"Điểm hiện tại: {score:.4f}")
                print(f"Tham số tốt nhất cho đến nay:")
                if self.best_params:
                    for param, value in self.best_params.items():
                        print(f"  {param}: {value}")
                
                # Lưu lịch sử
                self.history.append({
                    'iteration': iteration + 1,
                    'score': score,
                    'params': params.copy(),
                    'is_best': score == self.best_score,
                    'best_score': self.best_score
                })
                    
            except Exception as e:
                print(f"Lỗi trong lần lặp {iteration + 1}: {str(e)}")
                continue
        
        print("\n" + "=" * 50)
        print("Tối ưu hóa hoàn thành!")
        
        if self.best_params is not None:
            print("\nGiải pháp tốt nhất tìm được:")
            print(f"Điểm: {self.best_score:.4f}")
            print("Tham số:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        else:
            print("Không tìm thấy tham số hợp lệ!")
            
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):

        if self.best_params is None:
            print("Không có mô hình tối ưu nào!")
            return None
        
        # Tạo và huấn luyện mô hình cuối cùng
        final_rf = RandomForestClassifier(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            min_samples_leaf=self.best_params['min_samples_leaf'],
            max_features=self.best_params['max_features'],
            class_weight='balanced',
            n_jobs=-1
        )
        
        final_rf.fit(self.X_train_scaled, self.y_train)
        
        # Đưa ra dự đoán
        y_pred = final_rf.predict(self.X_test_scaled)
        y_prob = final_rf.predict_proba(self.X_test_scaled)
        
        # Lấy xác suất cho lớp 1
        if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        # Tính toán các chỉ số
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print("\nChỉ số Tập Kiểm tra:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Độ chính xác: {test_accuracy:.4f}")
        
        # Lấy tên đặc trưng
        feature_names = getattr(self, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        return {
            'model': final_rf,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'best_params': self.best_params,
            'feature_importances': dict(zip(feature_names, final_rf.feature_importances_))
        }

def main():
    """Hàm chính"""
    print("Đọc dữ liệu từ file Excel...")
    
    # Thay đổi đường dẫn này thành file dữ liệu của bạn
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Đã đọc {len(df)} hàng dữ liệu")
        
        # Cột đặc trưng (điều chỉnh theo file Excel của bạn)
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Cột nhãn (điều chỉnh theo file Excel của bạn)
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
        
        # Xử lý giá trị thiếu
        if np.isnan(X).any():
            print("CẢNH BÁO: Tìm thấy giá trị thiếu trong dữ liệu!")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        print(f"Hình dạng đặc trưng: {X.shape}")
        print("Phân bố nhãn:")
        # Chuyển đổi y thành numpy array và đảm bảo kiểu dữ liệu integer
        y_array = np.asarray(y, dtype=int)
        unique_labels = np.unique(y_array)
        label_counts = np.bincount(y_array)
        for label, count in zip(unique_labels, label_counts):
            print(f"  Lớp {label}: {count}")
        
        # Khởi tạo bộ tối ưu hóa Random Search với tên đặc trưng
        optimizer = RandomSearchOptimizer(X, y, n_iter=10, feature_names=feature_columns)
        
        # Chạy tối ưu hóa
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nThời gian tối ưu hóa: {end_time - start_time:.2f} giây")
        
        if best_params is not None:
            print("\nTham số tốt nhất:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"\nĐiểm tốt nhất: {best_score:.4f}")
            
            # Đánh giá mô hình cuối cùng
            print("\nĐánh giá mô hình trên tập kiểm tra:")
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print(f"F1-Score Kiểm tra: {final_results['test_f1']:.4f}")
                print(f"AUC Kiểm tra: {final_results['test_auc']:.4f}")
                print(f"Độ chính xác Kiểm tra: {final_results['test_accuracy']:.4f}")
                
                # # Lưu mô hình
                # joblib.dump(final_results['model'], 'best_flood_rf_randomsearch.pkl')
                # print("\nMô hình đã được lưu vào 'best_flood_rf_randomsearch.pkl'")
        else:
            print("\nTối ưu hóa thất bại trong việc tìm tham số hợp lệ.")
    
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()