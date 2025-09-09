import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from config import MODEL_DIR
import warnings
warnings.filterwarnings('ignore')

class FloodRegressor:
    def __init__(self):
        self.models = {}
        self.model_names = []
    
    def load_model(self, model_path, model_name=None):

        try:
            model = joblib.load(model_path)
            
            if model_name is None:
                # Lấy tên từ filename và loại bỏ các prefix
                filename = Path(model_path).stem
                # Loại bỏ các prefix thường gặp
                for prefix in ['model_', 'pso_', 'puma_', 'rso_', 'flood_']:
                    filename = filename.replace(prefix, '')
                model_name = filename
            
            self.models[model_name] = model
            self.model_names.append(model_name)
            
            print(f"✅ Đã tải '{model_name}' từ {Path(model_path).name}")
            return True
            
        except Exception as e:
            print(f"Lỗi tải mô hình {Path(model_path).name}: {e}")
            return False
    
    def load_all_models(self, models_dir=None):

        if models_dir is None:
            models_dir = MODEL_DIR
            
        models_path = Path(models_dir)
        
        if not models_path.exists():
            print(f"Thư mục {models_dir} không tồn tại")
            return False
        
        # Tìm tất cả file .joblib trong thư mục
        model_files = list(models_path.glob('*.joblib'))
        
        if not model_files:
            print(f"Không tìm thấy file .joblib nào trong {models_dir}")
            return False
        
        print(f"🔍 Tìm thấy {len(model_files)} file mô hình trong {models_dir}")
        
        successful_loads = 0
        for model_file in model_files:
            if self.load_model(model_file):
                successful_loads += 1
        
        print(f"📊 Đã tải thành công {successful_loads}/{len(model_files)} mô hình")
        
        if successful_loads == 0:
            return False
            
        return True
    
    def predict_regression(self, X, model_name):

        if model_name not in self.models:
            raise ValueError(f"Mô hình '{model_name}' chưa được tải")
        
        model = self.models[model_name]
        
        try:
            predictions = model.predict(X)
            
            # Đảm bảo output trong khoảng [0, 1]
            predictions = np.clip(predictions, 0, 1)
            
            print(f"   📈 {model_name}: {predictions.min():.4f} - {predictions.max():.4f}")
            return predictions
            
        except Exception as e:
            print(f"   ❌ Lỗi dự đoán {model_name}: {e}")
            return np.full(len(X), np.nan)
    
    def predict_all_models(self, X):
        
        if not self.models:
            raise ValueError("Chưa có mô hình nào được tải")

        results = {}
        
        print(f"🤖 Đang dự đoán xác suất với {len(self.models)} mô hình...")
        
        for model_name in self.model_names:
            print(f"  🔄 Dự đoán với '{model_name}'...")
            predictions = self.predict_regression(X, model_name)
            results[f'prob_{model_name}'] = predictions
        
        print("✅ Hoàn thành dự đoán regression!")
        return results
    
    def get_model_info(self):
        """Trả về thông tin các mô hình đã tải"""
        return {
            'total_models': len(self.models),
            'model_names': self.model_names.copy(),
            'models_loaded': list(self.models.keys())
        }

def predict_flood_probabilities(X, position_data, models_dir=None):

    if models_dir is None:
        models_dir = MODEL_DIR
    
    print(f"🎯 Dữ liệu đầu vào: {X.shape[0]:,} điểm, {X.shape[1]} features")
    
    # Khởi tạo regressor và tải models
    regressor = FloodRegressor()
    
    if not regressor.load_all_models(models_dir):
        raise ValueError(f"Không thể tải mô hình nào từ {models_dir}")

    # Dự đoán với tất cả mô hình
    predictions = regressor.predict_all_models(X)
    
    # Tạo DataFrame kết quả
    results = position_data.copy()
    
    # Thêm predictions (xác suất regression)
    for model_name, probs in predictions.items():
        results[model_name] = probs
    
    # Tính xác suất ensemble (trung bình)
    prob_cols = [col for col in results.columns if col.startswith('prob_')]
    if prob_cols:
        results['prob_ensemble'] = results[prob_cols].mean(axis=1)
    
    # Thống kê
    print(f"\n{'='*60}")
    print("📈 THỐNG KÊ XÁC SUẤT LŨ LỤT (REGRESSION)")
    print(f"{'='*60}")
    print(f"📍 Tổng điểm: {len(results):,}")
    print(f"🤖 Số mô hình: {len(prob_cols)}")
    
    if 'prob_ensemble' in results:
        ensemble = results['prob_ensemble']
        print(f"🎯 Xác suất ensemble:")
        print(f"   Trung bình: {ensemble.mean():.4f}")
        print(f"   Min-Max: {ensemble.min():.4f} - {ensemble.max():.4f}")
        
        # Phân loại rủi ro
        very_high = (ensemble >= 0.8).sum()
        high = ((ensemble >= 0.6) & (ensemble < 0.8)).sum()
        medium = ((ensemble >= 0.4) & (ensemble < 0.6)).sum()
        low = (ensemble < 0.4).sum()
        total = len(results)
        
        print(f"🔴 Rủi ro rất cao (≥0.8): {very_high:,} ({very_high/total*100:.1f}%)")
        print(f"🟡 Rủi ro cao (0.6-0.8): {high:,} ({high/total*100:.1f}%)")
        print(f"🟢 Rủi ro TB (0.4-0.6): {medium:,} ({medium/total*100:.1f}%)")
        print(f"🔵 Rủi ro thấp (<0.4): {low:,} ({low/total*100:.1f}%)")
    
    return results