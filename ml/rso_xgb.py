import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time

class XGBRandomizedSearch:
    def __init__(self, X, y, n_iterations=50):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = -np.inf
        
        # Chia dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y
        )
        
        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Phạm vi tham số XGBoost - Bộ tham số mở rộng
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
    
    def create_random_params(self):
        """Tạo bộ tham số ngẫu nhiên"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                params[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                # Phân phối log-uniform cho các tham số như C và gamma
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                params[param] = 10 ** random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                params[param] = random.choice(range_info['options'])
            elif range_info['type'] == 'tuple_choice':
                params[param] = random.choice(range_info['options'])
        return params
    
    def evaluate_params(self, params):
        """Đánh giá bộ tham số bằng cách sử dụng kiểm định chéo"""
        try:
            # Tạo dictionary chỉ chứa các tham số hợp lệ cho XGBClassifier
            model_params = {}
            valid_params = [
                'n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
                'reg_alpha', 'reg_lambda', 'min_child_weight', 'gamma',
                'max_delta_step', 'scale_pos_weight'
            ]
            
            for param in valid_params:
                if param in params:
                    model_params[param] = params[param]
            
            model = xgb.XGBClassifier(
                **model_params,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            
            # Kiểm định chéo
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1')
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Lỗi khi đánh giá tham số: {str(e)}")
            return -np.inf
    
    def search(self):
        """Thuật toán tìm kiếm ngẫu nhiên chính"""
        print("Bắt đầu Tìm kiếm Ngẫu nhiên XGBoost...")
        print(f"Dữ liệu: {len(self.X)} điểm, {self.X.shape[1]} đặc trưng")
        print(f"Số lần lặp: {self.n_iterations}")
        
        # Phân bố lớp
        unique_labels = np.unique(self.y)
        label_counts = np.bincount(self.y.astype(int))
        print("Phân bố lớp:")
        for label, count in zip(unique_labels, label_counts):
            print(f"  Lớp {label}: {count}")
        print("-" * 50)
        
        # Vòng lặp tìm kiếm ngẫu nhiên
        for iteration in range(self.n_iterations):
            print(f"\nLần lặp {iteration + 1}/{self.n_iterations}")
            
            # Tạo tham số ngẫu nhiên
            params = self.create_random_params()
            
            # Đánh giá tham số
            score = self.evaluate_params(params)
            
            # Cập nhật tốt nhất nếu cải thiện
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print("*** TÌM THẤY TỐT NHẤT MỚI! ***")
            
            # In kết quả
            print(f"Điểm hiện tại: {score:.4f}")
            print(f"Điểm tốt nhất cho đến nay: {self.best_score:.4f}")
            print("Tham số hiện tại:")
            for param, value in params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
        
        print("\n" + "=" * 50)
        print("Tìm kiếm Ngẫu nhiên hoàn thành!")
        if self.best_params is not None:
            print(f"\nĐiểm tốt nhất: {self.best_score:.4f}")
            print("Tham số tốt nhất:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """Đánh giá mô hình cuối cùng trên tập kiểm tra"""
        if self.best_params is None:
            print("Không có mô hình tối ưu nào!")
            return None
        
        # Tạo dictionary chỉ chứa các tham số hợp lệ cho XGBClassifier
        model_params = {}
        valid_params = [
            'n_estimators', 'max_depth', 'learning_rate', 'subsample', 
            'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
            'reg_alpha', 'reg_lambda', 'min_child_weight', 'gamma',
            'max_delta_step', 'scale_pos_weight'
        ]
        
        for param in valid_params:
            if param in self.best_params:
                model_params[param] = self.best_params[param]
        
        # Huấn luyện mô hình với tham số tốt nhất
        best_model = xgb.XGBClassifier(
            **model_params,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        best_model.fit(self.X_train_scaled, self.y_train)
        
        # Dự đoán trên tập kiểm tra
        y_pred = best_model.predict(self.X_test_scaled)
        y_prob = best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Tính toán các chỉ số
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print("\nKết quả Tập Kiểm tra:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Độ chính xác: {test_acc:.4f}")
        
        return {
            'model': best_model,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_params
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
        
        # Xử lý giá trị thiếu
        if np.isnan(X).any():
            print("CẢNH BÁO: Tìm thấy giá trị thiếu trong dữ liệu!")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Khởi tạo và chạy tìm kiếm ngẫu nhiên
        searcher = XGBRandomizedSearch(X, y, n_iterations=30)
        
        start_time = time.time()
        best_params, best_score = searcher.search()
        end_time = time.time()
        
        print(f"\nThời gian tìm kiếm: {end_time - start_time:.2f} giây")
        
        if best_params is not None:
            # Đánh giá mô hình cuối cùng
            print("\nĐánh giá mô hình cuối cùng trên tập kiểm tra:")
            final_results = searcher.evaluate_final_model()
            
            if final_results:
                print(f"\nKết quả Kiểm tra Cuối cùng:")
                print(f"F1-Score: {final_results['test_f1']:.4f}")
                print(f"AUC: {final_results['test_auc']:.4f}")
                print(f"Độ chính xác: {final_results['test_accuracy']:.4f}")
        else:
            print("\nTìm kiếm thất bại trong việc tìm tham số hợp lệ.")
    
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        print("Vui lòng đảm bảo file Excel của bạn tồn tại tại đường dẫn đã chỉ định")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()