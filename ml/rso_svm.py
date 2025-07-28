import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time

class SVMRandomizedSearch:
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
        
        # Điều chỉnh tham số dựa trên lựa chọn kernel
        kernel = params['kernel']
        if kernel == 'linear':
            # Kernel tuyến tính không sử dụng gamma, degree, coef0
            params.pop('gamma', None)
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'poly':
            # Kernel đa thức sử dụng tất cả tham số
            pass
        elif kernel == 'rbf':
            # Kernel RBF không sử dụng degree, coef0
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'sigmoid':
            # Kernel sigmoid không sử dụng degree
            params.pop('degree', None)
        
        return params
    
    def evaluate_params(self, params):
        """Đánh giá bộ tham số bằng cách sử dụng kiểm định chéo"""
        try:
            # Tạo dictionary chỉ chứa các tham số hợp lệ cho SVC
            model_params = {}
            valid_params = [
                'C', 'gamma', 'kernel', 'degree', 'coef0', 'tol',
                'class_weight', 'max_iter', 'shrinking', 'probability'
            ]
            
            for param in valid_params:
                if param in params:
                    model_params[param] = params[param]
            
            # Thiết lập giá trị mặc định cho các tham số quan trọng
            if 'probability' not in model_params:
                model_params['probability'] = True  # Bật ước lượng xác suất cho tính toán AUC
            if 'max_iter' not in model_params:
                model_params['max_iter'] = 10000  # Tăng số lần lặp tối đa để đảm bảo hội tụ
            
            model = SVC(
                **model_params,
                random_state=42
            )
            
            # Cross-validation với bảo vệ timeout
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1', n_jobs=1)
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Lỗi khi đánh giá tham số: {str(e)}")
            return -np.inf
    
    def search(self):
        """Thuật toán tìm kiếm ngẫu nhiên chính"""
        print("Bắt đầu Tìm kiếm Ngẫu nhiên SVM...")
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
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        print("\n" + "=" * 50)
        print("Tìm kiếm Ngẫu nhiên hoàn thành!")
        if self.best_params is not None:
            print(f"\nĐiểm tốt nhất: {self.best_score:.4f}")
            print("Tham số tốt nhất:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """Đánh giá mô hình cuối cùng trên tập kiểm tra"""
        if self.best_params is None:
            print("Không có mô hình tối ưu nào!")
            return None
        
        # Huấn luyện mô hình với tham số tốt nhất
        best_model = SVC(
            C=self.best_params.get('C', 1.0),
            gamma=self.best_params.get('gamma', 'scale'),
            kernel=self.best_params.get('kernel', 'rbf'),
            degree=self.best_params.get('degree', 3),
            coef0=self.best_params.get('coef0', 0.0),
            class_weight=self.best_params.get('class_weight', None),
            tol=self.best_params.get('tol', 1e-3),
            probability=True,
            max_iter=10000
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
        
        # In thông tin support vectors
        print(f"Số lượng support vectors: {best_model.n_support_}")
        print(f"Support vectors mỗi lớp: {dict(zip(best_model.classes_, best_model.n_support_))}")
        
        return {
            'model': best_model,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': {
                'C': self.best_params['C'],
                'kernel': self.best_params['kernel'],
                'gamma': self.best_params['gamma']
            },
            'n_support_vectors': best_model.n_support_
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
        
        # Kiểm tra kích thước dữ liệu cho hiệu quả SVM
        if len(X) > 10000:
            print(f"CẢNH BÁO: Tập dữ liệu lớn ({len(X)} mẫu). SVM có thể chậm.")
            print("Hãy xem xét sử dụng một tập con hoặc chuyển sang kernel tuyến tính.")
        
        # Khởi tạo và chạy tìm kiếm ngẫu nhiên
        searcher = SVMRandomizedSearch(X, y, n_iterations=30)
        
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
                
                # Thông tin cụ thể về SVM
                print(f"\nThông tin Mô hình:")
                print(f"Kernel tốt nhất: {final_results['best_params']['kernel']}")
                print(f"Tổng support vectors: {sum(final_results['n_support_vectors'])}")
        else:
            print("\nTìm kiếm thất bại trong việc tìm tham số hợp lệ.")
    
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        print("Vui lòng đảm bảo file Excel của bạn tồn tại tại đường dẫn đã chỉ định")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()