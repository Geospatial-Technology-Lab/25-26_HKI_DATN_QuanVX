import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from typing import Dict, Union

def evaluate_regression_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray, 
                            clip_predictions: bool = True, return_detailed: bool = False) -> Union[float, Dict[str, float]]:

    try:
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = model.predict(X_test)
        if clip_predictions:
            y_pred = np.clip(y_pred, 0, 1)  # Giới hạn dự đoán từ 0 đến 1
        
        y_test_array = np.array(y_test)
        
        # Tính toán metrics
        r2 = r2_score(y_test_array, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
        mae = mean_absolute_error(y_test_array, y_pred)
        
        r2_positive = max(0, r2)  # Đảm bảo R² không âm
        fitness = r2_positive - rmse - mae
        
        if return_detailed:
            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'fitness': fitness,
                'fitness_score': float(fitness)  # Điểm dương cho thuật toán tối đa hóa
            }
        else:
            # Trả về điểm dương để tối đa hóa (cho thuật toán tối ưu)
            return float(fitness)
        
    except Exception as e:
        print(f"Lỗi trong đánh giá regression: {str(e)}")
        if return_detailed:
            return {
                'r2': np.nan,
                'rmse': np.inf,
                'mae': np.inf,
                'fitness': -np.inf,  # Điểm thấp nhất cho tối đa hóa
                'fitness_score': -np.inf
            }
        else:
            return -np.inf  # Trả về giá trị thấp nhất cho tối đa hóa khi lỗi