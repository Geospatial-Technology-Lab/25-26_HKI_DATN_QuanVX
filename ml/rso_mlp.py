import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import random
import time
import warnings
warnings.filterwarnings('ignore')

class MLPRandomizedSearch:
    def __init__(self, X, y, n_iterations=50):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = -np.inf
        
        # Split data (deterministic split - no randomness)
        n_samples = len(self.X)
        split_idx = int(0.8 * n_samples)  # 80% train, 20% test
        
        self.X_train = self.X[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_train = self.y[:split_idx]
        self.y_test = self.y[split_idx:]
        
        # Scale data (rất quan trọng cho MLP)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Chuẩn hóa target cho regression
        self.y_scaler = StandardScaler()
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test_scaled = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()
        
        # MLP parameter ranges
        self.param_ranges = {
            'hidden_layer_sizes': {
                'type': 'choice',
                'choices': [
                    (50,), (100,), (150,), (200,),
                    (50, 50), (100, 50), (150, 100), (200, 100),
                    (100, 100), (150, 150), (200, 150),
                    (50, 50, 50), (100, 50, 50), (100, 100, 50),
                    (150, 100, 50), (200, 100, 50)
                ]
            },
            'activation': {
                'type': 'choice', 
                'choices': ['relu', 'tanh', 'logistic']
            },
            'solver': {
                'type': 'choice',
                'choices': ['adam', 'lbfgs', 'sgd']
            },
            'alpha': {'type': 'float', 'min': 1e-5, 'max': 1e-1},
            'learning_rate': {
                'type': 'choice',
                'choices': ['constant', 'invscaling', 'adaptive']
            },
            'learning_rate_init': {'type': 'float', 'min': 1e-4, 'max': 1e-1},
            'max_iter': {'type': 'int', 'min': 200, 'max': 1000},
            'beta_1': {'type': 'float', 'min': 0.8, 'max': 0.999},
            'beta_2': {'type': 'float', 'min': 0.9, 'max': 0.9999},
            'epsilon': {'type': 'float', 'min': 1e-9, 'max': 1e-6}
        }
    
    def create_random_params(self):
        """Create random parameter set - Giữ tính ngẫu nhiên cho việc tối ưu hóa"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                # Sử dụng log scale cho một số tham số
                if param in ['alpha', 'learning_rate_init', 'epsilon']:
                    log_min = np.log10(range_info['min'])
                    log_max = np.log10(range_info['max'])
                    params[param] = 10 ** random.uniform(log_min, log_max)
                else:
                    params[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'choice':
                params[param] = random.choice(range_info['choices'])
        
        # Điều chỉnh tham số dựa trên solver
        if params['solver'] == 'lbfgs':
            # lbfgs chỉ hoạt động tốt với dữ liệu nhỏ và ít layers
            if len(params['hidden_layer_sizes']) > 2:
                params['hidden_layer_sizes'] = random.choice([
                    (50,), (100,), (150,), (200,),
                    (50, 50), (100, 50), (100, 100)
                ])
            params['max_iter'] = random.randint(200, 500)  # Giảm max_iter cho lbfgs
        
        return params
    
    def evaluate_params(self, params):
        """Evaluate parameter set using cross-validation"""
        try:
            model = MLPRegressor(
                hidden_layer_sizes=params['hidden_layer_sizes'],
                activation=params['activation'],
                solver=params['solver'],
                alpha=params['alpha'],
                learning_rate=params['learning_rate'],
                learning_rate_init=params['learning_rate_init'],
                max_iter=params['max_iter'],
                beta_1=params['beta_1'],
                beta_2=params['beta_2'],
                epsilon=params['epsilon'],
                random_state=None,  # Loại bỏ random_state để có tính ngẫu nhiên trong training
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # Cross-validation với neg_mean_squared_error để có thể maximize
            # Sử dụng shuffle=False để có tính deterministic trong CV
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train_scaled, 
                                      cv=3, scoring='neg_mean_squared_error', 
                                      shuffle=False)
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return -np.inf
    
    def search(self):
        """Main randomized search algorithm - Tính ngẫu nhiên chỉ áp dụng cho việc tối ưu hóa tham số"""
        print("Starting MLP Randomized Search for Regression...")
        print("(Deterministic data split, randomized parameter optimization)")
        print(f"Data: {len(self.X)} points, {self.X.shape[1]} features")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Number of iterations: {self.n_iterations}")
        
        # Target statistics
        print(f"Target statistics:")
        print(f"  Mean: {np.mean(self.y):.4f}")
        print(f"  Std: {np.std(self.y):.4f}")
        print(f"  Min: {np.min(self.y):.4f}")
        print(f"  Max: {np.max(self.y):.4f}")
        print("-" * 50)
        
        # Random search loop
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            
            # Generate random parameters
            params = self.create_random_params()
            
            # Evaluate parameters
            score = self.evaluate_params(params)
            
            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print("*** NEW BEST FOUND! ***")
            
            # Print results
            print(f"Current score: {score:.4f}")
            print(f"Best score so far: {self.best_score:.4f}")
            print("Current parameters:")
            for param, value in params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        print("\n" + "=" * 50)
        print("Randomized Search completed!")
        if self.best_params is not None:
            print(f"\nBest score: {self.best_score:.4f}")
            print("Best parameters:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """Evaluate final model on test set"""
        if self.best_params is None:
            print("No optimized model available!")
            return None
        
        # Train model with best parameters
        best_model = MLPRegressor(
            hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
            activation=self.best_params['activation'],
            solver=self.best_params['solver'],
            alpha=self.best_params['alpha'],
            learning_rate=self.best_params['learning_rate'],
            learning_rate_init=self.best_params['learning_rate_init'],
            max_iter=self.best_params['max_iter'],
            beta_1=self.best_params['beta_1'],
            beta_2=self.best_params['beta_2'],
            epsilon=self.best_params['epsilon'],
            random_state=None,  # Loại bỏ random_state để có tính ngẫu nhiên trong training
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        best_model.fit(self.X_train_scaled, self.y_train_scaled)
        
        # Predict on test set
        y_pred_scaled = best_model.predict(self.X_test_scaled)
        
        # Transform predictions back to original scale
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate regression metrics
        test_mse = mean_squared_error(self.y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        
        print("\nTest Set Results:")
        print(f"MSE: {test_mse:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"R²: {test_r2:.4f}")
        
        # In thông tin mô hình
        print(f"\nModel Information:")
        print(f"Number of layers: {best_model.n_layers_}")
        print(f"Number of outputs: {best_model.n_outputs_}")
        print(f"Number of iterations: {best_model.n_iter_}")
        
        return {
            'model': best_model,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'best_params': self.best_params
        }

def main():
    """Main function"""
    print("Reading data from Excel file...")
    
    # Change this path to your data file
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Read {len(df)} rows of data")
        
        # Feature columns
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Target column (continuous value for regression)
        target_column = 'target_value'  # Giá trị liên tục cần dự đoán
        
        # Check for missing columns
        missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Following columns not found: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("WARNING: Missing values found in data!")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Initialize and run randomized search
        searcher = MLPRandomizedSearch(X, y, n_iterations=30)
        
        start_time = time.time()
        best_params, best_score = searcher.search()
        end_time = time.time()
        
        print(f"\nSearch time: {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            # Evaluate final model
            print("\nEvaluating final model on test set:")
            final_results = searcher.evaluate_final_model()
            
            if final_results:
                print(f"\nFinal Test Results:")
                print(f"MSE: {final_results['test_mse']:.4f}")
                print(f"RMSE: {final_results['test_rmse']:.4f}")
                print(f"MAE: {final_results['test_mae']:.4f}")
                print(f"R²: {final_results['test_r2']:.4f}")
        else:
            print("\nSearch failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
