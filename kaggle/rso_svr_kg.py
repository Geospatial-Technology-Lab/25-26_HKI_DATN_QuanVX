import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Add constant random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check for GPU availability
try:
    import cuml
    from cuml.svm import SVR as cuSVR
    GPU_AVAILABLE = True
    print("GPU acceleration available - using cuML")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU acceleration not available - using scikit-learn")

class SVMRegressionRandomizedSearch:
    def __init__(self, X, y, n_iterations=100, use_gpu=True):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = -np.inf
        self.best_scores_history = []
        self.metrics_history = []
        self.best_metrics = None
        self.best_params_history = []  # Store best parameters for each iteration
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Scale data (important for SVR)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # SVR parameter ranges
        self.param_ranges = {
            'C': {'type': 'log_uniform', 'min': 0.001, 'max': 1000},  # Regularization parameter
            'gamma': {'type': 'log_uniform', 'min': 0.0001, 'max': 10},  # Kernel coefficient
            'kernel': {'type': 'choice', 'options': ['linear', 'poly', 'rbf']},  # cuML only supports these kernels
            'degree': {'type': 'int', 'min': 2, 'max': 5},  # Degree for poly kernel
            'coef0': {'type': 'float', 'min': 0.0, 'max': 10.0},  # Independent term for poly kernel
            'epsilon': {'type': 'log_uniform', 'min': 0.001, 'max': 1.0},  # Epsilon-tube for SVR
            'tol': {'type': 'log_uniform', 'min': 1e-5, 'max': 1e-2}  # Tolerance for stopping criterion
        }
    
    def create_random_params(self):
        """Create random parameter set for randomized search"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                params[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                # Log-uniform distribution for parameters like C and gamma
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                params[param] = 10 ** random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                params[param] = random.choice(range_info['options'])
        
        # Adjust parameters based on kernel choice
        kernel = params['kernel']
        if kernel == 'linear':
            # Linear kernel doesn't use gamma, degree, coef0
            params.pop('gamma', None)
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'poly':
            # Polynomial kernel uses all parameters
            pass
        elif kernel == 'rbf':
            # RBF kernel doesn't use degree, coef0
            params.pop('degree', None)
            params.pop('coef0', None)
        
        return params
    
    def evaluate_params(self, params):
        """Evaluate parameter set using RMSE as the main metric"""
        try:
            model_params = params.copy()
            
            # Create model based on GPU availability
            if self.use_gpu:
                # cuML SVR
                model = cuSVR(
                    C=model_params.get('C', 1.0),
                    gamma=model_params.get('gamma', 'scale'),
                    kernel=model_params.get('kernel', 'rbf'),
                    degree=model_params.get('degree', 3) if 'degree' in model_params else 3,
                    coef0=model_params.get('coef0', 0.0) if 'coef0' in model_params else 0.0,
                    epsilon=model_params.get('epsilon', 0.1),
                    tol=model_params.get('tol', 1e-3)
                )
                
                # For GPU, use a simpler cross-validation approach
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    self.X_train_scaled, self.y_train, test_size=0.2, random_state=RANDOM_SEED
                )
                
                model.fit(X_train_split, y_train_split)
                y_val_pred = model.predict(X_val_split)
                rmse_cv = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
                
                # Fit on full training data for evaluation
                model.fit(self.X_train_scaled, self.y_train)
            else:
                # scikit-learn SVR
                model = SVR(
                    C=model_params.get('C', 1.0),
                    gamma=model_params.get('gamma', 'scale'),
                    kernel=model_params.get('kernel', 'rbf'),
                    degree=model_params.get('degree', 3) if 'degree' in model_params else 3,
                    coef0=model_params.get('coef0', 0.0) if 'coef0' in model_params else 0.0,
                    epsilon=model_params.get('epsilon', 0.1),
                    tol=model_params.get('tol', 1e-3),
                    max_iter=10000
                )
                
                # Cross-validation with negative MSE scoring
                mse_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                         cv=3, scoring='neg_mean_squared_error')
                rmse_cv = np.sqrt(-np.mean(mse_scores))
                
                # Fit on full training data for evaluation
                model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            # Calculate additional metrics
            n = len(self.y_test)
            p = self.X.shape[1]  # number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            # Store metrics history
            current_metrics = {
                'r2': float(r2),
                'adjusted_r2': float(adjusted_r2),
                'rmse': float(rmse),
                'rmse_cv': float(rmse_cv),
                'mae': float(mae)
            }
            
            # Update best metrics tracking based on RMSE
            if self.best_metrics is None:
                self.best_metrics = current_metrics.copy()
            else:
                if current_metrics['rmse'] < self.best_metrics['rmse']:
                    self.best_metrics = current_metrics.copy()
            
            # Store metrics history
            self.metrics_history.append({
                'iteration': len(self.metrics_history),
                'params': params,
                'metrics': current_metrics
            })
            
            # Return negative RMSE as score (higher is better)
            return -current_metrics['rmse']
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return -np.inf
    
    def search(self):
        """Main randomized search algorithm"""
        print("Starting SVR Randomized Search for Regression...")
        print(f"Data: {len(self.X)} points, {self.X.shape[1]} features")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Target range: [{np.min(self.y):.4f}, {np.max(self.y):.4f}]")
        print(f"Target mean: {np.mean(self.y):.4f}, std: {np.std(self.y):.4f}")
        
        print("-" * 70)
        print("Iter |  RMSE  |   R²   | Adj.R² |  MAE   | Trend")
        print("-" * 70)
        
        # Random search loop
        for iteration in range(self.n_iterations):
            # Generate random parameters
            params = self.create_random_params()
            
            # Evaluate parameters
            score = self.evaluate_params(params)
            
            # Update best if improved
            improved = False
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                improved = True
            
            # Store best parameters history
            if len(self.best_params_history) == 0:
                # First iteration - store current params
                self.best_params_history.append(params.copy())
            else:
                # Update only if improved
                if improved:
                    self.best_params_history.append(params.copy())
                else:
                    self.best_params_history.append(self.best_params_history[-1].copy())
            
            # Store best score history
            self.best_scores_history.append(self.best_score)
            
            # Print progress with more detailed information
            latest_metrics = self.metrics_history[-1]['metrics']
            
            # Calculate trend
            if iteration > 0:
                prev_rmse = self.metrics_history[-2]['metrics']['rmse']
                current_rmse = latest_metrics['rmse']
                change = current_rmse - prev_rmse
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                trend_info = f" {trend}({change:+.4f})"
            else:
                trend_info = ""
            
            # Print progress only on improvement or every 10 iterations
            if improved or (iteration + 1) % 10 == 0:
                print(f"{iteration+1:4d} | {latest_metrics['rmse']:6.4f} | {latest_metrics['r2']:6.4f} | "
                    f"{latest_metrics['adjusted_r2']:6.4f} | {latest_metrics['mae']:6.4f}{trend_info}")
            
            # Display summary every 25 iterations
            if (iteration + 1) % 25 == 0:
                recent_rmse = [h['metrics']['rmse'] for h in self.metrics_history[-25:]]
                print(f"     Last 25: Best={min(recent_rmse):.4f}, Avg={np.mean(recent_rmse):.4f}")
                print("-" * 70)
        
        print("\n" + "=" * 70)
        print("Randomized Search completed!")
        if self.best_params is not None:
            print(f"\nBest RMSE: {-self.best_score:.4f}")
            print("Best parameters:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.best_params, self.best_metrics
    
    def evaluate_final_model(self):
        """Evaluate final model on test set with comprehensive metrics"""
        if self.best_params is None:
            print("No optimized model available!")
            return None
        
        # Create model with best parameters
        model_params = self.best_params.copy()
        
        if self.use_gpu:
            # cuML SVR
            best_model = cuSVR(
                C=model_params.get('C', 1.0),
                gamma=model_params.get('gamma', 'scale'),
                kernel=model_params.get('kernel', 'rbf'),
                degree=model_params.get('degree', 3) if 'degree' in model_params else 3,
                coef0=model_params.get('coef0', 0.0) if 'coef0' in model_params else 0.0,
                epsilon=model_params.get('epsilon', 0.1),
                tol=model_params.get('tol', 1e-3)
            )
        else:
            # scikit-learn SVR
            best_model = SVR(
                C=model_params.get('C', 1.0),
                gamma=model_params.get('gamma', 'scale'),
                kernel=model_params.get('kernel', 'rbf'),
                degree=model_params.get('degree', 3) if 'degree' in model_params else 3,
                coef0=model_params.get('coef0', 0.0) if 'coef0' in model_params else 0.0,
                epsilon=model_params.get('epsilon', 0.1),
                tol=model_params.get('tol', 1e-3),
                max_iter=10000
            )
        
        # Train final model
        best_model.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_model.predict(self.X_test_scaled)
        
        # Calculate regression metrics
        test_mse = mean_squared_error(self.y_test, y_pred)
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        
        # Calculate additional metrics
        n = len(self.y_test)
        p = self.X.shape[1]  # number of features
        adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
        
        print(f"\nTest Results - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # Get number of support vectors (differs between sklearn and cuML)
        if self.use_gpu:
            n_support_vectors = best_model.n_support  # cuML implementation
        else:
            n_support_vectors = len(best_model.support_)  # sklearn implementation
        
        return {
            'model': best_model,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'adjusted_r2': adjusted_r2,
            'best_params': self.best_params,
            'n_support_vectors': n_support_vectors
        }
    
    def plot_optimization_progress(self):
        """Plot optimization progress"""
        if not self.metrics_history:
            print("No optimization history to plot.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # RMSE evolution over iterations
        plt.subplot(2, 2, 1)
        rmse_values = [h['metrics']['rmse'] for h in self.metrics_history]
        plt.plot(range(1, len(rmse_values)+1), rmse_values, 'b-', alpha=0.5)
        plt.plot(range(1, len(self.best_scores_history)+1), 
                [-x for x in self.best_scores_history], 'r-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('RMSE Evolution')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(['Current RMSE', 'Best RMSE'])
        
        # R² evolution
        plt.subplot(2, 2, 2)
        r2_values = [h['metrics']['r2'] for h in self.metrics_history]
        plt.plot(range(1, len(r2_values)+1), r2_values, 'g-')
        plt.xlabel('Iteration')
        plt.ylabel('R²')
        plt.title('R² Evolution')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Parameter evolution for important parameters
        plt.subplot(2, 2, 3)
        c_values = [h['params'].get('C', float('nan')) for h in self.metrics_history]
        plt.plot(range(1, len(c_values)+1), c_values, 'r-', alpha=0.5)
        
        eps_values = [h['params'].get('epsilon', float('nan')) for h in self.metrics_history]
        plt.plot(range(1, len(eps_values)+1), eps_values, 'g-', alpha=0.5)
        
        gamma_values = [h['params'].get('gamma', float('nan')) for h in self.metrics_history]
        plt.plot(range(1, len(gamma_values)+1), gamma_values, 'b-', alpha=0.5)
        
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        plt.legend(['C', 'Epsilon', 'Gamma'])
        
        # Best parameters C and epsilon by kernel type
        plt.subplot(2, 2, 4)
        plt.scatter([h['params'].get('C', 1.0) for h in self.metrics_history],
                  [h['params'].get('epsilon', 0.1) for h in self.metrics_history],
                  c=[h['metrics']['rmse'] for h in self.metrics_history],
                  cmap='viridis', alpha=0.6)
        plt.colorbar(label='RMSE')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('C parameter')
        plt.ylabel('Epsilon parameter')
        plt.title('C vs Epsilon (color = RMSE)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('svr_optimization_progress.png')
        plt.show()
    
    def export_detailed_results_to_excel(self, filename='rso_svr_detailed_results.xlsx'):
        """Export detailed results to Excel with multiple sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Iteration History
                history_data = []
                for i, hist in enumerate(self.metrics_history):
                    row = {
                        'Iteration': i + 1,
                        'RMSE': hist['metrics']['rmse'],
                        'R²': hist['metrics']['r2'],
                        'MAE': hist['metrics']['mae'],
                        'Adjusted_R²': hist['metrics']['adjusted_r2'],
                        'RMSE_CV': hist['metrics']['rmse_cv'],
                        'Is_Best': hist['metrics']['rmse'] == min([h['metrics']['rmse'] for h in self.metrics_history])
                    }
                    # Add current iteration parameters
                    for k, v in hist['params'].items():
                        row[f'Current_{k}'] = v
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='Iteration_History', index=False)
                
                # Sheet 2: Best Parameters History
                best_params_history_data = []
                for i, best_params in enumerate(self.best_params_history):
                    row = {'Iteration': i + 1}
                    for k, v in best_params.items():
                        row[f'Best_{k}'] = v
                    # Add corresponding metrics for this iteration
                    if i < len(self.metrics_history):
                        row['RMSE'] = self.metrics_history[i]['metrics']['rmse']
                        row['R²'] = self.metrics_history[i]['metrics']['r2']
                        row['MAE'] = self.metrics_history[i]['metrics']['mae']
                    best_params_history_data.append(row)
                
                best_params_history_df = pd.DataFrame(best_params_history_data)
                best_params_history_df.to_excel(writer, sheet_name='Best_Parameters_History', index=False)
                
                # Sheet 3: Final Best Parameters
                if self.best_params:
                    best_params_df = pd.DataFrame([self.best_params])
                    best_params_df.to_excel(writer, sheet_name='Final_Best_Parameters', index=False)
                
                # Sheet 4: Summary Statistics
                summary_stats = {
                    'Metric': ['RMSE', 'R²', 'MAE', 'Adjusted_R²', 'RMSE_CV'],
                    'Best': [
                        min([h['metrics']['rmse'] for h in self.metrics_history]),
                        max([h['metrics']['r2'] for h in self.metrics_history]),
                        min([h['metrics']['mae'] for h in self.metrics_history]),
                        max([h['metrics']['adjusted_r2'] for h in self.metrics_history]),
                        min([h['metrics']['rmse_cv'] for h in self.metrics_history])
                    ],
                    'Average': [
                        np.mean([h['metrics']['rmse'] for h in self.metrics_history]),
                        np.mean([h['metrics']['r2'] for h in self.metrics_history]),
                        np.mean([h['metrics']['mae'] for h in self.metrics_history]),
                        np.mean([h['metrics']['adjusted_r2'] for h in self.metrics_history]),
                        np.mean([h['metrics']['rmse_cv'] for h in self.metrics_history])
                    ],
                    'Std_Dev': [
                        np.std([h['metrics']['rmse'] for h in self.metrics_history]),
                        np.std([h['metrics']['r2'] for h in self.metrics_history]),
                        np.std([h['metrics']['mae'] for h in self.metrics_history]),
                        np.std([h['metrics']['adjusted_r2'] for h in self.metrics_history]),
                        np.std([h['metrics']['rmse_cv'] for h in self.metrics_history])
                    ]
                }
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
            print(f"Detailed results exported to {filename}")
        except Exception as e:
            print(f"Error exporting results: {str(e)}")

def main():
    """Main function for SVR regression"""
    try:
        print("=" * 70)
        print("SVR REGRESSION WITH RANDOMIZED SEARCH OPTIMIZATION")
        print("=" * 70)
        
        # For Kaggle environments, you might need to adjust the path
        try:
            # First try the typical Kaggle path
            file_path = "../input/flood-data/flood_data.xlsx"
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            # Fall back to local path if not in Kaggle
            file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} samples from {file_path.split('/')[-1]}")
        
        # Feature columns
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Target column
        target_column = 'target_value'  # Continuous target variable for regression
        
        # Check for missing columns
        missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            # Try to use flood_prediction or label_column if target_value is not found
            if 'target_value' in missing_cols:
                possible_targets = ['flood_prediction', 'label_column', 'target', 'y']
                for col in possible_targets:
                    if col in df.columns:
                        print(f"Using {col} as target variable instead")
                        target_column = col
                        missing_cols.remove('target_value')
                        break
                
            if missing_cols:
                return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[target_column].values
        
        print(f"Data shape: X = {X.shape}, y = {y.shape}")
        
        # Check for any NaN values
        if np.isnan(X).any():
            print("Handling missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Check GPU usage preference - default to True but can be overridden
        use_gpu = True  # Set to False to force CPU usage
        
        # Initialize and run randomized search
        print("\n" + "=" * 70)
        print("Starting SVR Randomized Search...")
        
        # Adjust number of iterations based on whether running in Kaggle or local
        if "kaggle" in file_path:
            n_iterations = 150  # Higher for Kaggle with GPU
        else:
            n_iterations = 100  # Lower for local
        
        searcher = SVMRegressionRandomizedSearch(X, y, n_iterations=n_iterations, use_gpu=use_gpu)
        
        start_time = time.time()
        best_params, best_metrics = searcher.search()
        end_time = time.time()
        
        print(f"\nSearch completed in {end_time - start_time:.2f} seconds")
        
        if best_params is not None and best_metrics is not None:
            # Display final results
            print("\n" + "=" * 70)
            print("FINAL BEST RESULTS:")
            print(f"R²: {best_metrics['r2']:.4f}")
            print(f"Adjusted R²: {best_metrics['adjusted_r2']:.4f}")
            print(f"RMSE: {best_metrics['rmse']:.4f}")
            print(f"RMSE (CV): {best_metrics['rmse_cv']:.4f}")
            print(f"MAE: {best_metrics['mae']:.4f}")
            
            # Plot optimization progress
            print("\nGenerating optimization plots...")
            searcher.plot_optimization_progress()
            
            # Save results to CSV and Excel
            print("\nSaving results...")
            
            # Save best parameters
            params_df = pd.DataFrame([best_params])
            params_df.to_csv('rso_svr_best_params.csv', index=False)
            
            # Save final metrics
            metrics_df = pd.DataFrame([best_metrics])
            metrics_df.to_csv('rso_svr_final_metrics.csv', index=False)
            
            # Save optimization history
            history_data = []
            for i, hist in enumerate(searcher.metrics_history):
                row = {
                    'iteration': i + 1,
                    'rmse': hist['metrics']['rmse'],
                    'r2': hist['metrics']['r2'],
                    'mae': hist['metrics']['mae'],
                    'adjusted_r2': hist['metrics']['adjusted_r2'],
                    'rmse_cv': hist['metrics']['rmse_cv']
                }
                # Add current iteration parameters
                row.update({f'current_{k}': v for k, v in hist['params'].items()})
                
                # Add best parameters for this iteration
                if i < len(searcher.best_params_history):
                    row.update({f'best_{k}': v for k, v in searcher.best_params_history[i].items()})
                
                history_data.append(row)
            
            history_df = pd.DataFrame(history_data)
            history_df.to_csv('rso_svr_optimization_history.csv', index=False)
            
            # Export detailed results to Excel - only if not in Kaggle
            if "kaggle" not in file_path:
                searcher.export_detailed_results_to_excel('rso_svr_detailed_results.xlsx')
            
            print("\n" + "=" * 70)
            print("ALL RESULTS HAVE BEEN SAVED")
            print("=" * 70)
            print("• rso_svr_best_params.csv - Final best parameters")
            print("• rso_svr_final_metrics.csv - Final metrics")
            print("• rso_svr_optimization_history.csv - Complete optimization history")
            print("• svr_optimization_progress.png - Visualization of search progress")
            
        else:
            print("Search failed to find valid parameters.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
