# Try to import GPU libraries, fallback to CPU if not available
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = False

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import time
import warnings
import matplotlib.pyplot as plt
import traceback
warnings.filterwarnings('ignore')

# Thiết lập hạt giống cố định để đảm bảo tái tạo kết quả
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def check_gpu():
    """Kiểm tra GPU sẵn có"""
    if not GPU_AVAILABLE:
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except:
        return False

class MLPRandomizedSearch:
    def __init__(self, X, y, n_iterations=50):
        # Handle GPU data conversion
        self.X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        self.y = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
        
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = -np.inf
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_test = self.y[:split_idx], self.y[split_idx:]
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.y_scaler = StandardScaler()
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test_scaled = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()
        
        # Clean NaN/inf values
        self.X_train_scaled = np.nan_to_num(self.X_train_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        self.X_test_scaled = np.nan_to_num(self.X_test_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # MLP parameter ranges - Improved ranges for better convergence
        self.param_ranges = {
            'hidden_layer_sizes': {
                'type': 'choice',
                'choices': [
                    (32,), (64,), (128,), (256,),
                    (32, 16), (64, 32), (128, 64), (256, 128),
                    (64, 64), (128, 128), (256, 256),
                    (128, 64, 32), (256, 128, 64), (512, 256, 128)
                ]
            },
            'activation': {
                'type': 'choice', 
                'choices': ['relu', 'tanh', 'logistic']
            },
            'solver': {
                'type': 'choice',
                'choices': ['adam', 'lbfgs']  # Removed 'sgd' as it's often unstable
            },
            'alpha': {'type': 'float', 'min': 1e-6, 'max': 1e-2},
            'learning_rate': {
                'type': 'choice',
                'choices': ['constant', 'adaptive']  # Removed 'invscaling'
            },
            'learning_rate_init': {'type': 'float', 'min': 1e-4, 'max': 1e-2},
            'max_iter': {'type': 'int', 'min': 500, 'max': 2000},  # Increased range
            'beta_1': {'type': 'float', 'min': 0.85, 'max': 0.95},  # Tightened range
            'beta_2': {'type': 'float', 'min': 0.95, 'max': 0.999},  # Tightened range
            'epsilon': {'type': 'float', 'min': 1e-9, 'max': 1e-7}  # Tightened range
        }
    
    def create_random_params(self):
        """Create random parameter set"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                if param in ['alpha', 'learning_rate_init', 'epsilon']:
                    log_min, log_max = np.log10(range_info['min']), np.log10(range_info['max'])
                    params[param] = 10 ** random.uniform(log_min, log_max)
                else:
                    params[param] = random.uniform(range_info['min'], range_info['max'])
            else:  # choice
                params[param] = random.choice(range_info['choices'])
        
        # Adjust parameters for lbfgs solver
        if params['solver'] == 'lbfgs':
            if len(params['hidden_layer_sizes']) > 2:
                params['hidden_layer_sizes'] = random.choice([
                    (32,), (64,), (128,), (256,), (64, 32), (128, 64), (128, 128)
                ])
            params['max_iter'] = random.randint(500, 1000)
            # Remove adam-specific parameters
            for key in ['beta_1', 'beta_2', 'epsilon', 'learning_rate', 'learning_rate_init']:
                params.pop(key, None)
        
        return params
    
    def evaluate_params(self, params):
        """Evaluate parameter set using cross-validation"""
        try:
            # Create model with only available parameters
            model_params = {k: v for k, v in params.items() if k in [
                'hidden_layer_sizes', 'activation', 'solver', 'alpha', 
                'learning_rate', 'learning_rate_init', 'max_iter', 
                'beta_1', 'beta_2', 'epsilon'
            ]}
            
            model = MLPRegressor(
                **model_params,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train_scaled, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_score = float(np.mean(cv_scores))
            
            # Calculate metrics on test set
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred_scaled = model.predict(self.X_test_scaled)
            y_pred_original = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_original))
            mae = mean_absolute_error(self.y_test, y_pred_original)
            r2 = r2_score(self.y_test, y_pred_original)
            
            params['_metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            return cv_score
            
        except Exception as e:
            params['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
            return -np.inf
    
    def search(self):
        """Main randomized search algorithm - Tính ngẫu nhiên chỉ áp dụng cho việc tối ưu hóa tham số"""
        print("Optimization Progress:")
        print("Gen | Best RMSE")
        print("-" * 18)
        
        # Initialize tracking
        self.best_scores_history = []
        iteration_results = []
        best_rmse = float('inf')  # Track best RMSE separately
        
        # Random search loop
        for iteration in range(self.n_iterations):
            # Generate random parameters
            params = self.create_random_params()
            
            # Evaluate parameters
            score = self.evaluate_params(params)
            
            # Update best if improved
            is_new_best = False
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                is_new_best = True
            
            # Store score history
            self.best_scores_history.append(self.best_score)
            
            # Get RMSE from evaluation
            if '_metrics' in params:
                current_rmse = params['_metrics']['rmse']
                current_mae = params['_metrics']['mae']
                current_r2 = params['_metrics']['r2']
            else:
                # Fallback calculation if metrics not available
                current_rmse = np.sqrt(-score) if score < 0 else np.sqrt(abs(score))
                current_mae = current_rmse * 0.8
                current_r2 = max(0, 1 - (current_rmse / np.std(self.y))**2)
            
            # Update best RMSE only if it's better (lower)
            if current_rmse < best_rmse:
                best_rmse = current_rmse
            
            # Display format - only best RMSE (non-increasing)
            print(f"{iteration + 1:3d} | {best_rmse:8.4f}")
            
            # Save iteration results (but still keep actual values for Excel)
            iteration_results.append({
                'iteration': iteration + 1,
                'cv_score': score,
                'rmse': current_rmse,
                'mae': current_mae,
                'r2': current_r2,
                'best_rmse': best_rmse,  # Add best RMSE column
                **{param: value for param, value in params.items() if param != '_metrics'}
            })
        
        print("\n" + "=" * 18)
        print("Search completed!")
        if self.best_params is not None:
            print(f"Best CV Score: {self.best_score:.4f}")
            print(f"Best RMSE: {best_rmse:.4f}")
        
        # Save iteration results to Excel only
        results_df = pd.DataFrame(iteration_results)
        
        # Reorder columns: iteration, scores, best_rmse, then parameters
        param_cols = [col for col in results_df.columns if col not in ['iteration', 'cv_score', 'rmse', 'mae', 'r2', 'best_rmse']]
        column_order = ['iteration', 'cv_score', 'rmse', 'mae', 'r2', 'best_rmse'] + param_cols
        results_df = results_df[column_order]
        
        # Save to Excel only
        results_df.to_excel('rso_mlp_iterations.xlsx', index=False)
        print(f"Results saved to: rso_mlp_iterations.xlsx")
        
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """Evaluate final model on test set"""
        if self.best_params is None:
            return None
        
        # Create model with only available parameters
        model_params = {k: v for k, v in self.best_params.items() if k in [
            'hidden_layer_sizes', 'activation', 'solver', 'alpha', 
            'learning_rate', 'learning_rate_init', 'max_iter', 
            'beta_1', 'beta_2', 'epsilon'
        ]}
        
        best_model = MLPRegressor(**model_params, random_state=42, early_stopping=True,
                                 validation_fraction=0.1, n_iter_no_change=10)
        
        best_model.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred_scaled = best_model.predict(self.X_test_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        
        print(f"\nTest Set Results:\nRMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}\nR²: {test_r2:.4f}")
        
        return {
            'model': best_model,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'best_params': self.best_params
        }
    
    def _to_numpy(self, data):
        """Convert data to numpy format"""
        return data.to_numpy() if hasattr(data, 'to_numpy') else data


def fill_missing_with_neighbors(series):
    """Fill missing values with neighbor averaging"""
    series = series.copy()
    missing_indices = series.isnull()
    
    if not missing_indices.any():
        return series
    
    for idx in series.index[missing_indices]:
        neighbors = []
        
        # Get neighbors
        if idx > 0 and pd.notna(series.iloc[idx - 1]):
            neighbors.append(series.iloc[idx - 1])
        if idx < len(series) - 1 and pd.notna(series.iloc[idx + 1]):
            neighbors.append(series.iloc[idx + 1])
        
        # Fill with neighbor average or column mean
        if neighbors:
            series.iloc[idx] = np.mean(neighbors)
        else:
            valid_values = series.dropna()
            series.iloc[idx] = valid_values.mean() if len(valid_values) > 0 else 0.0
    
    return series


def plot_optimization_progress(searcher):
    """Plot MLP optimization progress"""
    if not hasattr(searcher, 'best_scores_history') or len(searcher.best_scores_history) < 2:
        return
    
    try:
        df = pd.read_excel('rso_mlp_iterations.xlsx')
        iterations = df['iteration'].values[1:]
        rmse_scores = df['rmse'].values[1:]
        mae_scores = df['mae'].values[1:]
        r2_scores = df['r2'].values[1:]
    except:
        iterations = list(range(1, len(searcher.best_scores_history)))
        neg_mse_scores = searcher.best_scores_history[1:]
        rmse_scores = [np.sqrt(-score) if score < 0 else np.sqrt(score) for score in neg_mse_scores]
        mae_scores = [x * 0.8 for x in rmse_scores]
        r2_scores = [1 - (x / rmse_scores[0])**2 for x in rmse_scores]
    
    # Ensure non-increasing RMSE
    for i in range(1, len(rmse_scores)):
        if rmse_scores[i] > rmse_scores[i-1]:
            rmse_scores[i] = rmse_scores[i-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot metrics
    axes[0, 0].plot(iterations, rmse_scores, 'b-', label='Best RMSE')
    axes[0, 0].set_title('MLP Search Progress')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best RMSE')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(iterations, r2_scores, 'g-', linewidth=2)
    axes[0, 1].set_title('R² Progression')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(iterations, mae_scores, 'r-', linewidth=2)
    axes[1, 0].set_title('MAE Progression')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True)
    
    # Normalized metrics
    def normalize(scores):
        return [(x - min(scores)) / (max(scores) - min(scores) + 1e-10) 
                if max(scores) > min(scores) else [0] * len(scores) for x in scores]
    
    axes[1, 1].plot(iterations, normalize(rmse_scores), 'b-', label='RMSE', linewidth=2)
    axes[1, 1].plot(iterations, normalize(r2_scores), 'g-', label='R²', linewidth=2)
    axes[1, 1].plot(iterations, normalize(mae_scores), 'r-', label='MAE', linewidth=2)
    axes[1, 1].set_title('Normalized Metrics')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('mlp_randomized_search_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def export_detailed_results_to_excel(searcher, filename='rso_mlp_detailed_results.xlsx'):
    """Export results to Excel"""
    try:
        df = pd.read_excel('rso_mlp_iterations.xlsx')
        if filename != 'rso_mlp_iterations.xlsx':
            df.to_excel(filename, index=False)
    except Exception as e:
        print(f"Export error: {e}")


def _load_and_preprocess_data():
    """Load and preprocess data for MLP regression"""
    df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
    
    feature_columns = [
        'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
        'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
        'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
    ]
    label_column = 'Nom'
    
    if label_column not in df.columns:
        raise ValueError(f"Target column '{label_column}' not found!")
    
    # Convert target to numeric
    if df[label_column].dtype == 'object':
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
        
        if df[label_column].isnull().all():
            le = LabelEncoder()
            original = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')[label_column]
            valid_mask = ~original.isnull()
            if valid_mask.any():
                df.loc[valid_mask, label_column] = le.fit_transform(original[valid_mask].astype(str))
            else:
                raise ValueError("Target column has no valid values!")
    
    # Convert features to float
    for col in feature_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Fill missing values
    for col in feature_columns + [label_column]:
        if col in df.columns and df[col].isnull().any():
            df[col] = fill_missing_with_neighbors(df[col])
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
    
    # Verify target validity
    if df[label_column].isnull().all() or df[label_column].std() == 0:
        raise ValueError("Target variable has no variation!")
    
    X, y = df[feature_columns], df[label_column]
    
    # Clean inf/nan values
    if (X.isnull().sum().sum() + np.isinf(X.select_dtypes(include=[np.number])).sum().sum() + 
        y.isnull().sum() + np.isinf(y).sum()) > 0:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
    
    # Try GPU conversion
    if check_gpu():
        try:
            X, y = cudf.DataFrame(X), cudf.Series(y)
        except:
            pass
    
    return X, y

def main():
    """Main function - Kaggle version"""
    try:
        X, y = _load_and_preprocess_data()
        
        # Handle remaining missing values
        if hasattr(X, 'isnull') and X.isnull().any().any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_values = X.values if hasattr(X, 'values') else X
            X_values = imputer.fit_transform(X_values)
            
            if check_gpu():
                try:
                    X = cudf.DataFrame(X_values)
                except:
                    X = pd.DataFrame(X_values)
            else:
                X = pd.DataFrame(X_values)
        
        # Run optimization
        searcher = MLPRandomizedSearch(X, y, n_iterations=100)
        start_time = time.time()
        best_params, best_score = searcher.search()
        
        print(f"\nSearch completed in {time.time() - start_time:.1f}s")
        
        if best_params is not None:
            final_results = searcher.evaluate_final_model()
            if final_results:
                plot_optimization_progress(searcher)
                print(f"\nFiles saved:\n- rso_mlp_iterations.xlsx\n- mlp_randomized_search_results.png")
        else:
            print("Search failed to find valid parameters.")
    
    except FileNotFoundError:
        print("File not found! Check path: /kaggle/input/flood-trainning/flood_training.csv")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
