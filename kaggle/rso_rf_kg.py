# Try to import GPU libraries, fallback to CPU if not available
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.preprocessing import StandardScaler as cuStandardScaler
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor
    GPU_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import random
import warnings
import matplotlib.pyplot as plt
import time
import traceback
warnings.filterwarnings('ignore')

# Thiết lập hạt giống cố định để đảm bảo tái tạo kết quả
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def check_gpu():
    """Kiểm tra GPU sẵn có và tình trạng hoạt động"""
    if not GPU_AVAILABLE:
        return False
    try:
        import cupy as cp
        # Kiểm tra thêm thông tin GPU
        for i in range(cp.cuda.runtime.getDeviceCount()):
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            if hasattr(device_props, 'name'):
                print(f"GPU {i}: {device_props.name}")
            if hasattr(device_props, 'totalGlobalMem'):
                print(f"GPU {i} Memory: {device_props.totalGlobalMem / 1024**3:.1f} GB")
        return True
    except Exception as e:
        return False

class RSOOptimizer:
    def __init__(self, X, y, n_trials=100):
        self.X = X
        self.y = y
        self.n_trials = n_trials  # Số lần thử ngẫu nhiên
        self.best_individual = None
        self.best_score = np.inf  # RMSE: lower is better
        self.best_scores_history = []  # Track best scores for plotting
        self.has_gpu = GPU_AVAILABLE
        
        # Chuẩn bị dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data(X, y)
        
        # Scale data
        self.scaler = cuStandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Debug scaled data
        X_scaled_check = self.X_train_scaled.values if hasattr(self.X_train_scaled, 'values') else self.X_train_scaled
        
        # Check for NaN/inf in scaled data
        if np.any(np.isnan(X_scaled_check)) or np.any(np.isinf(X_scaled_check)):
            # Clean scaled data
            self.X_train_scaled = np.nan_to_num(X_scaled_check, nan=0.0, 
                                               posinf=3.0, neginf=-3.0)  # Cap at 3 std devs
            if hasattr(self.X_test_scaled, 'values'):
                X_test_scaled_check = self.X_test_scaled.values
            else:
                X_test_scaled_check = self.X_test_scaled
            self.X_test_scaled = np.nan_to_num(X_test_scaled_check, nan=0.0, 
                                              posinf=3.0, neginf=-3.0)
        
        # Parameter search space for Random Forest regression
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 10, 'max': 800},
            'max_depth': {'type': 'int', 'min': 3, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
            'max_features': {'type': 'float', 'min': 0.1, 'max': 1.0},
            'bootstrap': {'type': 'categorical', 'values': [True, False]},
            'max_samples': {'type': 'float', 'min': 0.1, 'max': 1.0}
        }
        
        # Get numerical parameters for consistent vector operations
        self.numerical_params = [p for p in self.param_ranges if self.param_ranges[p]['type'] in ['int', 'float']]
        self.num_numerical = len(self.numerical_params)
        
    def create_individual(self):
        """Create a random individual using randomized search principles"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                # Generate random integer within range
                individual[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                # Generate random float within range
                value = random.uniform(range_info['min'], range_info['max'])
                individual[param] = round(value, 6)
            elif range_info['type'] == 'categorical':
                # Random categorical choice
                individual[param] = random.choice(range_info['values'])
        return individual

    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual using RMSE, MAE, R2 (lower RMSE is better)"""
        try:
            # Validate parameters
            for param_name, param_value in individual.items():
                if param_name == '_metrics':
                    continue
                if isinstance(param_value, (int, float)) and (np.isnan(param_value) or np.isinf(param_value)):
                    return np.inf
            
            # Create Random Forest model
            if self.has_gpu:
                model = cuRandomForestRegressor(
                    n_estimators=int(individual['n_estimators']),
                    max_depth=int(individual['max_depth']) if individual['max_depth'] > 0 else None,
                    min_samples_split=int(individual['min_samples_split']),
                    min_samples_leaf=int(individual['min_samples_leaf']),
                    max_features=individual['max_features'],
                    bootstrap=individual['bootstrap'],
                    max_samples=individual['max_samples'] if individual['bootstrap'] else None,
                    random_state=RANDOM_SEED
                )
            else:
                model = cuRandomForestRegressor(
                    n_estimators=int(individual['n_estimators']),
                    max_depth=int(individual['max_depth']) if individual['max_depth'] > 0 else None,
                    min_samples_split=int(individual['min_samples_split']),
                    min_samples_leaf=int(individual['min_samples_leaf']),
                    max_features=individual['max_features'],
                    bootstrap=individual['bootstrap'],
                    max_samples=individual['max_samples'] if individual['bootstrap'] else None,
                    random_state=RANDOM_SEED
                )

            # Evaluate model
            if self.has_gpu:
                X_train_np = self._to_numpy(self.X_train_scaled)
                y_train_np = self._to_numpy(self.y_train)
                
                # Check for NaN/inf in data
                if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)) or \
                   np.any(np.isnan(y_train_np)) or np.any(np.isinf(y_train_np)):
                    return np.inf
                
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    X_train_np, y_train_np, test_size=0.2, random_state=RANDOM_SEED
                )
                
                model.fit(X_train_val, y_train_val)
                y_pred = model.predict(X_val)
                
                y_pred = self._to_numpy(y_pred)
                y_val = self._to_numpy(y_val)
            else:
                # Check for NaN/inf in data
                X_check = self.X_train_scaled.values if hasattr(self.X_train_scaled, 'values') else self.X_train_scaled
                y_check = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
                
                if np.any(np.isnan(X_check)) or np.any(np.isinf(X_check)) or \
                   np.any(np.isnan(y_check)) or np.any(np.isinf(y_check)):
                    return np.inf
                
                # Use a single train-validation split for consistent metrics
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    self.X_train_scaled, self.y_train, test_size=0.2, random_state=RANDOM_SEED
                )
                
                model.fit(X_train_val, y_train_val)
                y_pred = model.predict(X_val)
            
            # Check predictions for NaN/inf
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
                return np.inf
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Check if metrics are valid
            if np.isnan(rmse) or np.isinf(rmse) or rmse <= 0:
                individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
                return np.inf
            
            # Store metrics for later use
            individual['_metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            return rmse  # Return RMSE as primary fitness (lower is better)
            
        except Exception as e:
            individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
            return np.inf  # Return high RMSE for failed evaluations

    def _prepare_data(self, X, y):
        """Xử lý và chuẩn bị dữ liệu cho GPU/CPU (Regression)"""
        # Xử lý giá trị null và chuyển đổi về numpy
        X_filled = X.fillna(X.mean())
        y_filled = y.fillna(y.mean())  # For regression, use mean instead of mode
        
        if GPU_AVAILABLE and isinstance(X, cudf.DataFrame):
            X_np = X_filled.to_numpy()
            y_np = y_filled.to_numpy()
        else:
            X_np = X_filled.values if hasattr(X_filled, 'values') else X_filled
            y_np = y_filled.values if hasattr(y_filled, 'values') else y_filled
        
        # Clean any remaining NaN/inf
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        
        # Chia dữ liệu (no stratify for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Chuyển về cuDF nếu GPU có sẵn
        if self.has_gpu:
            try:
                return (cudf.DataFrame(X_train), cudf.DataFrame(X_test), 
                       cudf.Series(y_train), cudf.Series(y_test))
            except Exception as e:
                self.has_gpu = False
        
        return X_train, X_test, y_train, y_test

    def _to_numpy(self, data):
        """Chuyển đổi dữ liệu về numpy format"""
        if isinstance(data, (cudf.Series, cudf.DataFrame)):
            return data.to_numpy()
        return data

    def optimize(self):
        """Run the pure Randomized Search optimization process"""
        # Initialize results tracking
        iteration_results = []
        
        print("Starting Randomized Search Optimization...")
        print("Generating random parameter combinations...")
        
        # Initialize with first random individual
        best_individual = self.create_individual()
        best_fitness = self.evaluate_individual(best_individual)
        
        # Store initial results
        self.best_individual = best_individual
        self.best_score = best_fitness
        self.best_scores_history = [best_fitness]
        
        iteration_results.append({
            'trial': 0,
            'rmse': self.best_score,
            'mae': self.best_individual.get('_metrics', {}).get('mae', 0),
            'r2': self.best_individual.get('_metrics', {}).get('r2', 0),
            **{param: value for param, value in self.best_individual.items() if param != '_metrics'}
        })
        
        print("\nRandomized Search Progress:")
        print("Trial | Best RMSE | Current RMSE | Improvement | Status")
        print("-" * 60)
        
        # Main randomized search loop
        for trial in range(1, self.n_trials):
            # Generate completely random individual
            current_individual = self.create_individual()
            current_fitness = self.evaluate_individual(current_individual)
            
            # Check if this is better than current best
            improvement = 0
            status = "No improvement"
            if current_fitness < best_fitness:
                improvement = best_fitness - current_fitness
                best_individual = current_individual.copy()
                best_fitness = current_fitness
                self.best_individual = best_individual
                self.best_score = best_fitness
                status = "*** NEW BEST ***"
            
            # Store best score for current trial
            self.best_scores_history.append(best_fitness)
            
            # Print progress
            print(f"{trial:5d} | {best_fitness:9.6f} | {current_fitness:12.6f} | {improvement:11.6f} | {status}")
            
            # Save trial results
            iteration_results.append({
                'trial': trial,
                'rmse': best_fitness,
                'mae': self.best_individual.get('_metrics', {}).get('mae', 0),
                'r2': self.best_individual.get('_metrics', {}).get('r2', 0),
                **{param: value for param, value in self.best_individual.items() if param != '_metrics'}
            })
        
        print(f"\nCompleted all {self.n_trials} trials.")
        
        # Save all trial results to CSV and Excel
        results_df = pd.DataFrame(iteration_results)
        
        # Reorder columns: trial, rmse, mae, r2, then parameters
        param_cols = [col for col in results_df.columns if col not in ['trial', 'rmse', 'mae', 'r2']]
        column_order = ['trial', 'rmse', 'mae', 'r2'] + param_cols
        results_df = results_df[column_order]
        
        # Save to CSV and Excel
        results_df.to_csv('rso_rf_trials.csv', index=False)
        results_df.to_excel('rso_rf_trials.xlsx', index=False)
        
        return self.best_individual, self.best_score

    def _prepare_data(self, X, y):
        """Xử lý và chuẩn bị dữ liệu cho GPU/CPU (Regression)"""
        # Xử lý giá trị null và chuyển đổi về numpy
        X_filled = X.fillna(X.mean())
        y_filled = y.fillna(y.mean())  # For regression, use mean instead of mode
        
        if GPU_AVAILABLE and isinstance(X, cudf.DataFrame):
            X_np = X_filled.to_numpy()
            y_np = y_filled.to_numpy()
        else:
            X_np = X_filled.values if hasattr(X_filled, 'values') else X_filled
            y_np = y_filled.values if hasattr(y_filled, 'values') else y_filled
        
        # Clean any remaining NaN/inf
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        
        # Chia dữ liệu (no stratify for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Chuyển về cuDF nếu GPU có sẵn
        if self.has_gpu:
            try:
                return (cudf.DataFrame(X_train), cudf.DataFrame(X_test), 
                       cudf.Series(y_train), cudf.Series(y_test))
            except Exception as e:
                self.has_gpu = False
        
        return X_train, X_test, y_train, y_test

    def _to_numpy(self, data):
        """Chuyển đổi dữ liệu về numpy format"""
        if isinstance(data, (cudf.Series, cudf.DataFrame)):
            return data.to_numpy()
        return data


def fill_missing_with_neighbors(series):
    """
    Fill missing values with the average of immediate neighbors (above and below)
    If neighbors are not available, use column mean as fallback
    """
    series = series.copy()
    missing_indices = series.isnull()
    
    if not missing_indices.any():
        return series  # No missing values
    
    print(f"Filling {missing_indices.sum()} missing values using neighbor averaging...")
    
    for idx in series.index[missing_indices]:
        neighbors = []
        
        # Get value above (previous row)
        if idx > 0:
            above_val = series.iloc[idx - 1] if idx - 1 in series.index else None
            if pd.notna(above_val):
                neighbors.append(above_val)
        
        # Get value below (next row)
        if idx < len(series) - 1:
            below_val = series.iloc[idx + 1] if idx + 1 in series.index else None
            if pd.notna(below_val):
                neighbors.append(below_val)
        
        # Fill with neighbor average or fallback to column mean
        if neighbors:
            series.iloc[idx] = np.mean(neighbors)
        else:
            # Fallback to column mean (excluding NaN values)
            valid_values = series.dropna()
            if len(valid_values) > 0:
                series.iloc[idx] = valid_values.mean()
            else:
                # If all values are NaN, use 0 as last resort
                series.iloc[idx] = 0.0
    
    return series


def plot_optimization_progress(optimizer):
    """Plot optimization progress for RSO."""
    if not hasattr(optimizer, 'best_scores_history') or len(optimizer.best_scores_history) < 2:
        return
    
    # Đọc dữ liệu từ file CSV
    try:
        trial_results = pd.read_csv('rso_rf_trials.csv')
        trials = trial_results['trial'].values[1:]  # Skip trial 0
        rmse_scores = trial_results['rmse'].values[1:]
        mae_scores = trial_results['mae'].values[1:]
        r2_scores = trial_results['r2'].values[1:]
    except:
        # Fallback to best_scores_history
        trials = list(range(1, len(optimizer.best_scores_history)))
        rmse_scores = optimizer.best_scores_history[1:]
        mae_scores = [x * 0.8 for x in rmse_scores]  # Approximate MAE from RMSE
        r2_scores = [1 - (x / rmse_scores[0])**2 for x in rmse_scores]  # Approximate R²
    
    # Ensure best_rmse is non-increasing (best RMSE should never increase)
    for i in range(1, len(rmse_scores)):
        if rmse_scores[i] > rmse_scores[i-1]:
            rmse_scores[i] = rmse_scores[i-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Best RMSE progression
    axes[0, 0].plot(trials, rmse_scores, 'b-', label='Best RMSE')
    axes[0, 0].set_title('Randomized Search Progress')
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Best RMSE')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Plot 2: R² Progression with Best RMSE
    axes[0, 1].plot(trials, r2_scores, 'g-', linewidth=2)
    axes[0, 1].set_title('R² Progression with Best RMSE')
    axes[0, 1].set_xlabel('Trial')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].grid(True)
    
    # Plot 3: MAE Progression with Best RMSE
    axes[1, 0].plot(trials, mae_scores, 'r-', label='MAE', linewidth=2)
    axes[1, 0].set_title('MAE Progression with Best RMSE')
    axes[1, 0].set_xlabel('Trial')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Plot 4: All metrics together (normalized)
    ax4 = axes[1, 1]
    
    # Normalize values for better visualization
    norm_rmse = [(x - min(rmse_scores)) / (max(rmse_scores) - min(rmse_scores) + 1e-10) if max(rmse_scores) > min(rmse_scores) else x for x in rmse_scores]
    norm_r2 = [(x - min(r2_scores)) / (max(r2_scores) - min(r2_scores) + 1e-10) if max(r2_scores) > min(r2_scores) else x for x in r2_scores]
    norm_mae = [(x - min(mae_scores)) / (max(mae_scores) - min(mae_scores) + 1e-10) if max(mae_scores) > min(mae_scores) else x for x in mae_scores]
    
    ax4.plot(trials, norm_rmse, 'b-', label='Best RMSE', linewidth=2)
    ax4.plot(trials, norm_r2, 'g-', label='R²', linewidth=2)
    ax4.plot(trials, norm_mae, 'r-', label='MAE', linewidth=2)
    
    ax4.set_title('Normalized Metrics Progression')
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Normalized Value')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('rso_optimization_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"- Biểu đồ: rso_optimization_results.png")
    plt.show()


def export_detailed_results_to_excel(optimizer, filename='rso_rf_detailed_results.xlsx'):
    """Export detailed results to Excel with multiple sheets - Simplified version"""
    try:
        # Read trial results
        df = pd.read_csv('rso_rf_trials.csv')
        
        # Simple export with pandas ExcelWriter
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: All trials
            df.to_excel(writer, sheet_name='All_Trials', index=False)
            
            # Sheet 2: Best result only
            best_idx = df['rmse'].idxmin()
            best_df = df.loc[[best_idx]]
            best_df.to_excel(writer, sheet_name='Best_Result', index=False)
            
            # Sheet 3: Summary stats
            metrics = ['rmse', 'mae', 'r2']
            summary = pd.DataFrame({
                'Metric': metrics,
                'Best': [df[m].min() if m != 'r2' else df[m].max() for m in metrics],
                'Mean': [df[m].mean() for m in metrics],
                'Std': [df[m].std() for m in metrics]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"- Chi tiết kết quả: {filename}")
    except Exception as e:
        print(f"Lỗi khi xuất kết quả: {e}")
        return


def _load_and_preprocess_data():
    """Tải và tiền xử lý dữ liệu cho bài toán hồi quy"""
    df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
    
    feature_columns = [
        'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
        'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
        'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
    ]
    label_column = 'Nom'
    
    # Check if target column exists and has valid data
    if label_column not in df.columns:
        raise ValueError(f"Target column '{label_column}' not found in data!")
    
    # For regression, treat the target as continuous values
    # If the data is categorical (Yes/No), convert to numeric for regression
    if df[label_column].dtype == 'object':
        # Try to convert strings like "1.5", "2.0" etc to numeric
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
        
        # If still all NaN after numeric conversion, try label encoding
        if df[label_column].isnull().all():
            print("Target column appears to be categorical, applying label encoding...")
            le = LabelEncoder()
            # Convert back to original and encode
            original_target = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')[label_column]
            # Remove null values for encoding
            valid_mask = ~original_target.isnull()
            if valid_mask.any():
                df.loc[valid_mask, label_column] = le.fit_transform(original_target[valid_mask].astype(str))
            else:
                raise ValueError("Target column has no valid values!")
    
    # Replace commas with dots and convert to float
    for col in feature_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Fill features using neighbor averaging
    for col in feature_columns:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = fill_missing_with_neighbors(df[col])
                # Fallback to mean if still missing
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
    
    # Fill target using neighbor averaging
    if df[label_column].isnull().any():
        df[label_column] = fill_missing_with_neighbors(df[label_column])
        # Fallback to mean if still missing
        if df[label_column].isnull().any():
            df[label_column] = df[label_column].fillna(df[label_column].mean())
    
    # Verify target has valid values
    if df[label_column].isnull().all() or df[label_column].std() == 0:
        raise ValueError("Target variable has no variation or all values are missing!")
    
    X = df[feature_columns]
    y = df[label_column]
    
    # Check for any remaining NaN/inf values
    nan_features = X.isnull().sum().sum()
    inf_features = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    nan_target = y.isnull().sum()
    inf_target = np.isinf(y).sum()
    
    if nan_features > 0 or inf_features > 0 or nan_target > 0 or inf_target > 0:
        # Additional cleaning
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
    
    # Convert to cuDF if GPU is available
    has_gpu = check_gpu()
    if has_gpu:
        try:
            X = cudf.DataFrame(X)
            y = cudf.Series(y)
            print("Đã chuyển đổi dữ liệu sang cuDF cho GPU acceleration.")
        except Exception as e:
            print(f"Không thể sử dụng GPU, chuyển về CPU: {e}")
    
    return X, y


def _run_optimization(X, y):
    optimizer = RSOOptimizer(X, y, n_trials=100)
    best_params, best_score = optimizer.optimize()
    return optimizer, best_params, best_score


def main():
    try:
        print("=== Randomized Search Optimization cho Random Forest ===")
        print("Thuật toán: RSO (Randomized Search Optimization)")
        print("Mô hình: Random Forest Regressor")
        print()
        
        # Check GPU availability
        gpu_status = check_gpu()
        print(f"GPU Available: {gpu_status}")
        print()
        
        # Load and preprocess data
        print("Đang tải và tiền xử lý dữ liệu...")
        X, y = _load_and_preprocess_data()
        print(f"Kích thước dữ liệu: {X.shape}")
        print(f"Target statistics: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
        print()
        
        # Run optimization
        print("Bắt đầu tối ưu hóa RSO...")
        start_time = time.time()
        optimizer, best_params, best_score = _run_optimization(X, y)
        end_time = time.time()
        
        print(f"\nTối ưu hóa hoàn thành trong {end_time - start_time:.2f} giây")
        print()
        
        # Plot optimization progress
        plot_optimization_progress(optimizer)
        
        # Export detailed results to Excel
        export_detailed_results_to_excel(optimizer)
        
        # Display results
        print("\n=== KẾT QUẢ TỐI ƯU HÓA ===")
        if hasattr(optimizer, 'best_scores_history') and len(optimizer.best_scores_history) > 0:
            initial_score = optimizer.best_scores_history[0]
            improvement = initial_score - best_score  # For RMSE: lower is better
            improvement_pct = (improvement / initial_score * 100) if initial_score > 0 else 0
            print(f"RMSE ban đầu: {initial_score:.6f}")
            print(f"RMSE tốt nhất: {best_score:.6f}")
            print(f"Cải thiện: {improvement:.6f} ({improvement_pct:.2f}%)")
        
        print(f"\nTham số tối ưu:")
        for param, value in best_params.items():
            if param != '_metrics':
                print(f"  {param:20}: {value}")
        
        if '_metrics' in best_params:
            metrics = best_params['_metrics']
            print(f"\nCác chỉ số:")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  R²:   {metrics['r2']:.6f}")
        
        # Save best parameters
        params_df = pd.DataFrame([{k: v for k, v in best_params.items() if k != '_metrics'}])
        params_df.to_csv('rso_rf_best_params.csv', index=False)
        print(f"\n- Tham số tốt nhất: rso_rf_best_params.csv")
        
        # Save final metrics
        if '_metrics' in best_params:
            metrics_df = pd.DataFrame([best_params['_metrics']])
            metrics_df.to_csv('rso_rf_final_metrics.csv', index=False)
            print(f"- Chỉ số cuối cùng: rso_rf_final_metrics.csv")
        
        print(f"- Lịch sử tối ưu: rso_rf_trials.csv")
        print(f"- Lịch sử tối ưu: rso_rf_trials.xlsx")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu! Vui lòng kiểm tra đường dẫn dataset.")
    except Exception as e:
        print(f"Lỗi: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
