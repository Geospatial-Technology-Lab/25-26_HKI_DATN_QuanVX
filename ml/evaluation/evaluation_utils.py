import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from typing import Dict, Union, Tuple

def evaluate_regression_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray, 
                            clip_predictions: bool = True, return_detailed: bool = False) -> Union[float, Dict[str, float]]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if clip_predictions:
        y_pred = np.clip(y_pred, 0, 1)
    y_test_array = np.array(y_test)
    r2 = r2_score(y_test_array, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
    mae = mean_absolute_error(y_test_array, y_pred)
    r2_positive = max(0, r2)
    fitness = r2_positive - rmse - mae
    if return_detailed:
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'fitness': fitness, 'fitness_score': float(fitness)}
    else:
        return float(fitness)

def load_data_from_csv(csv_file_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_file_path)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:14].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_data_info(csv_file_path: str) -> Dict[str, Union[int, float, str]]:
    data = pd.read_csv(csv_file_path)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:14]
    info = {
        'total_samples': data.shape[0],
        'total_columns': data.shape[1],
        'features_count': X.shape[1],
        'label_min': float(y.min()),
        'label_max': float(y.max()),
        'label_mean': float(y.mean()),
        'label_std': float(y.std()),
        'missing_values': data.isnull().sum().sum(),
        'file_path': csv_file_path
    }
    return info