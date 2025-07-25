# GPU is required and available by default
import cudf
import cupy as cp
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Add constant random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def check_gpu():
    """Check if GPU is available and ready to use"""
    try:
        import cudf
        import cupy as cp
        return True
    except ImportError:
        return False


class PSOMLPOptimizer:
    """Particle Swarm Optimization for MLP hyperparameter tuning for regression."""
    
    def __init__(self, X, y, n_particles=10, n_iterations=100):
        """Initialize PSO optimizer."""
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.has_gpu = check_gpu()
        
        # Prepare data
        self._prepare_data()
        
        # PSO parameters
        self.w = 0.9    # Inertia weight
        self.c1 = 1.5   # Cognitive parameter
        self.c2 = 1.5   # Social parameter
        self.w_min = 0.4 # Minimum inertia weight
        
        # Parameter search space for MLP regression
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
        
        # Get numerical and choice parameters for PSO operations
        self.numerical_params = [p for p in self.param_ranges if self.param_ranges[p]['type'] in ['int', 'float']]
        self.choice_params = [p for p in self.param_ranges if self.param_ranges[p]['type'] == 'choice']
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Optimization results
        self.global_best_position = {}
        self.global_best_rmse = np.inf
        self.optimization_history = []
        self.best_metrics = None
        self.metrics_history = []
    
    def _prepare_data(self):
        """Prepare and split data for training."""
        # Handle missing values
        if np.isnan(self.X).any():
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Scale features (MLP needs scaled features)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def _initialize_swarm(self):
        """Initialize particle swarm with random parameter generation."""
        # Initialize positions and velocities
        self.positions = [self._create_individual() for _ in range(self.n_particles)]
        
        self.velocities = []
        for i in range(self.n_particles):
            velocity = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'choice':
                    velocity[param] = 0  # No velocity for choice parameters
                else:
                    max_velocity = (range_info['max'] - range_info['min']) * 0.1
                    velocity[param] = np.random.uniform(-max_velocity, max_velocity)
            self.velocities.append(velocity)
        
        # Initialize personal best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_rmse = np.full(self.n_particles, np.inf)
    
    def _create_individual(self):
        """Create a random individual"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                # Random integer within range
                individual[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                # Random float within range
                individual[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'choice':
                # Random choice from choices
                individual[param] = random.choice(range_info['choices'])
        return individual
    
    def _evaluate_fitness(self, position):
        """Evaluate fitness using RMSE (lower is better)"""
        try:
            # Validate numerical parameters
            for param_name, param_value in position.items():
                if param_name == '_metrics':
                    continue
                if param_name not in ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate'] and (np.isnan(param_value) or np.isinf(param_value)):
                    return np.inf
            
            # Create MLP model parameters
            model_params = {
                'hidden_layer_sizes': position['hidden_layer_sizes'],
                'activation': position['activation'],
                'solver': position['solver'],
                'alpha': float(position['alpha']),
                'learning_rate': position['learning_rate'],
                'learning_rate_init': float(position['learning_rate_init']),
                'max_iter': int(position['max_iter']),
                'random_state': RANDOM_SEED,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'tol': 1e-4
            }
            
            # Add solver-specific parameters
            if position['solver'] == 'adam':
                model_params.update({
                    'beta_1': float(position['beta_1']),
                    'beta_2': float(position['beta_2']),
                    'epsilon': float(position['epsilon'])
                })

            model = MLPRegressor(**model_params)

            # Evaluate model with GPU support if available
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
                # Use train-validation split for evaluation
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    self.X_train_scaled, self.y_train, test_size=0.2, random_state=RANDOM_SEED
                )
                
                model.fit(X_train_val, y_train_val)
                y_pred = model.predict(X_val)
            
            # Check for invalid predictions
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return np.inf
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            if np.isnan(rmse) or np.isinf(rmse) or rmse <= 0:
                return np.inf
            
            # Store metrics
            position['_metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            
            # Update best metrics tracking
            current_metrics = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
            
            if self.best_metrics is None or current_metrics['rmse'] < self.best_metrics['rmse']:
                self.best_metrics = current_metrics.copy()
            
            # Store metrics history
            self.metrics_history.append({
                'iteration': len(self.metrics_history),
                'params': position,
                'metrics': current_metrics
            })
            
            return rmse
            
        except Exception as e:
            position['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
            return np.inf
    
    def _update_particle(self, particle_idx):
        """Update particle velocity and position."""
        w = self.w - (self.w - self.w_min) * (particle_idx / self.n_particles)
        
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'choice':
                # For choice parameters, probabilistic update
                r = np.random.random()
                if r < 0.3:  # Follow personal best
                    self.positions[particle_idx][param] = self.personal_best_positions[particle_idx][param]
                elif r < 0.6:  # Follow global best
                    self.positions[particle_idx][param] = self.global_best_position[param]
            else:
                # Standard PSO update for numerical parameters
                r1, r2 = np.random.random(2)
                cognitive = self.c1 * r1 * (self.personal_best_positions[particle_idx][param] - 
                                          self.positions[particle_idx][param])
                social = self.c2 * r2 * (self.global_best_position[param] - 
                                       self.positions[particle_idx][param])
                
                self.velocities[particle_idx][param] = (w * self.velocities[particle_idx][param] + 
                                                      cognitive + social)
                self.positions[particle_idx][param] += self.velocities[particle_idx][param]
                self.positions[particle_idx][param] = self._apply_bounds(param, 
                                                                        self.positions[particle_idx][param])
    
    def _apply_bounds(self, param, new_val):
        """Apply bounds constraints to parameter values."""
        range_info = self.param_ranges[param]
        
        if range_info['type'] == 'choice':
            return new_val
        
        # Clip to bounds and convert type
        clipped_val = np.clip(new_val, range_info['min'], range_info['max'])
        return int(round(clipped_val)) if range_info['type'] == 'int' else clipped_val
    
    def optimize(self):
        """Execute PSO optimization algorithm."""
        print("Iter | Best RMSE |   R²   |  MAE  ")
        print("-" * 40)
        
        # Evaluate initial swarm
        for i in range(self.n_particles):
            rmse = self._evaluate_fitness(self.positions[i])
            self.personal_best_rmse[i] = rmse
            
            if rmse < self.global_best_rmse:
                self.global_best_rmse = rmse
                self.global_best_position = self.positions[i].copy()
        
        # Store initial best position
        self.optimization_history.append({
            'iteration': 0,
            'best_rmse': self.global_best_rmse,
            'best_params': self.global_best_position.copy()
        })
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                self._update_particle(i)
                rmse = self._evaluate_fitness(self.positions[i])
                
                # Update personal and global best
                if rmse < self.personal_best_rmse[i]:
                    self.personal_best_rmse[i] = rmse
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                    if rmse < self.global_best_rmse:
                        self.global_best_rmse = rmse
                        self.global_best_position = self.positions[i].copy()

            # Print progress for this iteration
            if self.metrics_history:
                best_r2 = best_mae = 0
                for hist in self.metrics_history:
                    if abs(hist['metrics']['rmse'] - self.global_best_rmse) < 1e-6:
                        best_r2 = max(best_r2, hist['metrics']['r2'])
                        if best_mae == 0:
                            best_mae = hist['metrics']['mae']
                        else:
                            best_mae = min(best_mae, hist['metrics']['mae'])
                
                print(f"{iteration+1:4d} | {self.global_best_rmse:9.4f} | {best_r2:6.4f} | {best_mae:6.4f}")

            # Store history
            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_rmse': self.global_best_rmse,
                'best_params': self.global_best_position.copy()
            })
        
        return self.global_best_position, self.global_best_rmse
    
    def plot_optimization_progress(self):
        """Plot optimization progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Best RMSE progression
        iterations = range(1, len(self.optimization_history))  # Skip iteration 0
        best_rmse = [h['best_rmse'] for h in self.optimization_history[1:]]  # Skip first entry
        
        # Ensure best_rmse is non-increasing (best RMSE should never increase)
        for i in range(1, len(best_rmse)):
            if best_rmse[i] > best_rmse[i-1]:
                best_rmse[i] = best_rmse[i-1]
        
        axes[0, 0].plot(iterations, best_rmse, 'b-', label='Best RMSE')
        axes[0, 0].set_title('PSO MLP Optimization Progress')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best RMSE')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot 2: Find R² values corresponding to best RMSE
        best_r2_values = []
        for i in range(len(best_rmse)):
            # Find best R² for each best RMSE
            iter_best_r2 = 0
            for hist in self.metrics_history:
                if abs(hist['metrics']['rmse'] - best_rmse[i]) < 1e-6:
                    iter_best_r2 = max(iter_best_r2, hist['metrics']['r2'])
            best_r2_values.append(iter_best_r2)
            
        axes[0, 1].plot(iterations, best_r2_values, 'g-', linewidth=2)
        axes[0, 1].set_title('R² Progression with Best RMSE')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].grid(True)
        
        # Plot 3: Find MAE values corresponding to best RMSE
        best_mae_values = []
        for i in range(len(best_rmse)):
            # Find best MAE for each best RMSE
            iter_best_mae = float('inf')
            for hist in self.metrics_history:
                if abs(hist['metrics']['rmse'] - best_rmse[i]) < 1e-6:
                    iter_best_mae = min(iter_best_mae, hist['metrics']['mae'])
            best_mae_values.append(iter_best_mae)
        
        axes[1, 0].plot(iterations, best_mae_values, 'r-', label='MAE', linewidth=2)
        axes[1, 0].set_title('MAE Progression with Best RMSE')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot 4: All metrics together
        ax4 = axes[1, 1]
        
        # Normalize values for better visualization
        norm_rmse = [(x - min(best_rmse)) / (max(best_rmse) - min(best_rmse) + 1e-10) if max(best_rmse) > min(best_rmse) else x for x in best_rmse]
        norm_r2 = [(x - min(best_r2_values)) / (max(best_r2_values) - min(best_r2_values) + 1e-10) if max(best_r2_values) > min(best_r2_values) else x for x in best_r2_values]
        norm_mae = [(x - min(best_mae_values)) / (max(best_mae_values) - min(best_mae_values) + 1e-10) if max(best_mae_values) > min(best_mae_values) else x for x in best_mae_values]
        
        ax4.plot(iterations, norm_rmse, 'b-', label='Best RMSE', linewidth=2)
        ax4.plot(iterations, norm_r2, 'g-', label='R²', linewidth=2)
        ax4.plot(iterations, norm_mae, 'r-', label='MAE', linewidth=2)
        
        ax4.set_title('Normalized Metrics Progression')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Normalized Value')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

    def _to_numpy(self, data):
        """Convert data to numpy array, handling different data types"""
        if hasattr(data, 'values'):
            return data.values
        elif hasattr(data, 'to_numpy'):
            return data.to_numpy()
        elif hasattr(data, 'get'):
            return np.array(data.get())
        else:
            return np.array(data)
    
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
    
    # Feature columns for flood prediction
    feature_columns = [
        'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
        'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
        'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
    ]
    
    # Convert Yes/No to 1/0 for regression
    df['Nom'] = (df['Nom'] == 'Yes').astype(float)
    
    # Replace comma with dot and convert to float
    for col in feature_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Fill missing values
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
    
    X, y = df[feature_columns].values, df['Nom'].values
    
    # Handle remaining missing values
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    return X, y, feature_columns


def main():
    """Main execution function for PSO-MLP optimization"""
    # Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # Initialize and run optimizer
    optimizer = PSOMLPOptimizer(X=X, y=y, n_particles=10, n_iterations=100)
    
    # Optimize hyperparameters
    best_params, best_rmse = optimizer.optimize()
    
    # Display final results
    print("FINAL BEST RESULTS:")
    if optimizer.best_metrics:
        print(f"RMSE: {optimizer.best_metrics['rmse']:.4f}")
        print(f"MAE: {optimizer.best_metrics['mae']:.4f}")
        print(f"R²: {optimizer.best_metrics['r2']:.4f}")
    
    # Plot optimization progress
    optimizer.plot_optimization_progress()
    
    # Save optimization results to CSV
    history_data = []
    for i, hist in enumerate(optimizer.optimization_history[1:], 1):
        # Ensure best_rmse is non-increasing
        best_rmse = hist['best_rmse']
        if i > 1 and best_rmse > history_data[-1]['best_rmse']:
            best_rmse = history_data[-1]['best_rmse']
        
        # Find corresponding R² and MAE
        best_r2 = best_mae = 0
        for m_hist in optimizer.metrics_history:
            if abs(m_hist['metrics']['rmse'] - best_rmse) < 1e-6:
                best_r2 = max(best_r2, m_hist['metrics']['r2'])
                best_mae = min(best_mae, m_hist['metrics']['mae']) if best_mae == 0 else min(best_mae, m_hist['metrics']['mae'])
        
        row = {
            'iteration': hist['iteration'],
            'best_rmse': best_rmse,
            'best_r2': best_r2,
            'best_mae': best_mae
        }
        
        # Add best parameters
        for k, v in hist['best_params'].items():
            if k != '_metrics':
                row[f'best_{k}'] = v
        
        history_data.append(row)
    
    pd.DataFrame(history_data).to_csv('pso_mlp_optimization_results.csv', index=False)
    print("Results saved to pso_mlp_optimization_results.csv")


if __name__ == "__main__":
    main()