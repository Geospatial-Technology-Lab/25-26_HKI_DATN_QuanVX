"""
PUMA Optimizer for Random Forest - Kaggle Version with GPU Support
==================================================================
This script is optimized for running on Kaggle platform with GPU support.
Simplified fitness function using RMSE as primary metric with R² and MAE as supplementary metrics.
"""

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error, 
    mean_absolute_error
)
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings('ignore')

# Add constant random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def check_gpu():
    """Kiểm tra GPU sẵn có và tình trạng hoạt động"""
    if not GPU_AVAILABLE:
        print("GPU libraries not available, using CPU")
        return False
    try:
        import cupy as cp
        # Kiểm tra thêm thông tin GPU
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Found {device_count} GPU(s)")
        for i in range(device_count):
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            if hasattr(device_props, 'name'):
                print(f"GPU {i}: {device_props.name}")
            if hasattr(device_props, 'totalGlobalMem'):
                print(f"GPU {i} Memory: {device_props.totalGlobalMem / 1024**3:.1f} GB")
        return True
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False

class PUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        # Check GPU availability
        self.use_gpu = check_gpu()
        
        # Convert to appropriate format based on GPU availability
        if self.use_gpu and GPU_AVAILABLE:
            self.X = cudf.DataFrame(X) if not isinstance(X, cudf.DataFrame) else X
            self.y = cudf.Series(y) if not isinstance(y, cudf.Series) else y
        else:
            self.X = np.array(X)
            self.y = np.array(y)
            
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = np.inf  # Changed to inf since we're minimizing RMSE
        self.best_scores_history = []  # Track best scores for plotting
        self.metrics_history = []  # Track all metrics for each generation
        self.best_metrics = None  # Track best metrics
        self.pCR = 0.5  # Initial crossover rate
        self.p = 0.1    # pCR adjustment rate
        
        # Split and scale data with fixed random seed
        if self.use_gpu and GPU_AVAILABLE:
            # Convert to numpy for train_test_split, then back to cudf
            X_np = self.X.to_pandas().values if hasattr(self.X, 'to_pandas') else np.array(self.X)
            y_np = self.y.to_pandas().values if hasattr(self.y, 'to_pandas') else np.array(self.y)
        else:
            X_np = np.array(self.X)
            y_np = np.array(self.y)
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_np, y_np, test_size=0.2, stratify=y_np, random_state=RANDOM_SEED
        )
        
        # Scale data using appropriate scaler
        self.scaler = cuStandardScaler() if (self.use_gpu and GPU_AVAILABLE) else StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Parameter search space for Random Forest regression
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 10, 'max': 500},
            'max_depth': {'type': 'int', 'min': 3, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
            'max_features': {'type': 'float', 'min': 0.1, 'max': 1.0},
            'bootstrap': {'type': 'categorical', 'values': [True, False]},
            'max_samples': {'type': 'float', 'min': 0.1, 'max': 1.0}
        }
        
        # Get numerical parameters for consistent vector operations
        self.numerical_params = list(self.param_ranges.keys())
        self.num_numerical = len(self.numerical_params)
    
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        used_combinations = set()  # Track used combinations to avoid duplicates
        
        while True:
            temp_individual = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'int':
                    range_size = range_info['max'] - range_info['min']
                    # Add small random noise to avoid clustering around certain values
                    noise = random.gauss(0, range_size * 0.1)  # 10% of range as standard deviation
                    rand_val = random.random() * range_size + range_info['min'] + noise
                    # Ensure value stays within bounds after adding noise
                    rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                    temp_individual[param] = int(round(rand_val))
                elif range_info['type'] == 'float':
                    range_size = range_info['max'] - range_info['min']
                    # Add small random noise to avoid clustering around certain values
                    noise = random.gauss(0, range_size * 0.05)  # 5% of range for float parameters
                    rand_val = random.random() * range_size + range_info['min'] + noise
                    # Ensure value stays within bounds after adding noise
                    rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                    temp_individual[param] = round(rand_val, 3)  # Round to 3 decimal places
                elif range_info['type'] == 'categorical':
                    temp_individual[param] = random.choice(range_info['values'])
            
            # Create a tuple of parameters to check for duplicates
            param_tuple = tuple(temp_individual.values())
            if param_tuple not in used_combinations:
                individual = temp_individual
                used_combinations.add(param_tuple)
                break
            
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate fitness using RMSE as primary metric, with R² and MAE as supplementary metrics"""
        try:
            # Create Random Forest model using appropriate implementation
            if self.use_gpu and GPU_AVAILABLE:
                # For GPU version, use only supported parameters
                model = cuRandomForestRegressor(
                    n_estimators=individual['n_estimators'],
                    max_depth=individual['max_depth'],
                    min_samples_split=individual['min_samples_split'],
                    min_samples_leaf=individual['min_samples_leaf'],
                    random_state=RANDOM_SEED
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=individual['n_estimators'],
                    max_depth=individual['max_depth'],
                    min_samples_split=individual['min_samples_split'],
                    min_samples_leaf=individual['min_samples_leaf'],
                    max_features=individual['max_features'],
                    bootstrap=individual['bootstrap'],
                    max_samples=individual['max_samples'] if individual['bootstrap'] else None,
                    n_jobs=-1,
                    random_state=RANDOM_SEED
                )

            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Get predictions (flood probability from 0 to 1)
            y_pred = model.predict(self.X_test_scaled)
            
            # Convert cupy arrays to numpy if needed
            if self.use_gpu and GPU_AVAILABLE and hasattr(y_pred, 'get'):
                y_pred = y_pred.get()
                y_test = self.y_test.get() if hasattr(self.y_test, 'get') else self.y_test
            else:
                y_pred = np.array(y_pred)
                y_test = np.array(self.y_test)
            
            # Clip predictions to ensure they're between 0 and 1
            y_pred = np.clip(y_pred, 0, 1)
            
            # Calculate only 3 key metrics: R², MAE, RMSE
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store simplified metrics
            current_metrics = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
            
            # Initialize best_metrics if not set (using RMSE for comparison - lower is better)
            if self.best_metrics is None:
                self.best_metrics = current_metrics.copy()
                self.metrics_history.append({
                    'generation': len(self.metrics_history),
                    'params': individual,
                    'metrics': current_metrics
                })
            else:
                # Update based on RMSE (lower is better)
                if rmse < self.best_metrics['rmse']:
                    self.best_metrics = current_metrics.copy()
                    self.metrics_history.append({
                        'generation': len(self.metrics_history),
                        'params': individual,
                        'metrics': current_metrics
                    })
                else:
                    # Add entry with best metrics to maintain history length
                    self.metrics_history.append({
                        'generation': len(self.metrics_history),
                        'params': individual,
                        'metrics': self.best_metrics
                    })
            
            # Return RMSE as fitness (lower is better, so we'll negate it for maximization)
            return float(-rmse)  # Negative because optimization algorithms typically maximize
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return -np.inf
            
    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        used_combinations = set()  # Track parameter combinations across population
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions randomly
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Create new solution
            attempts = 0
            max_attempts = 10  # Limit attempts to avoid infinite loop
            while attempts < max_attempts:
                new_individual = current.copy()  # Start with current solution
                
                # Ensure at least one parameter changes by selecting random parameter
                j0 = random.choice(list(self.param_ranges.keys()))
                
                for param, range_info in self.param_ranges.items():
                    # Always change j0 parameter or based on pCR probability
                    if param == j0 or random.random() <= self.pCR:
                        if random.random() < 0.5:
                            if range_info['type'] == 'int':
                                # Generate random value with added noise for better diversity
                                range_size = range_info['max'] - range_info['min']
                                noise = random.gauss(0, range_size * 0.1)
                                rand_val = random.random() * range_size + range_info['min'] + noise
                                rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                                new_individual[param] = int(round(rand_val))
                            elif range_info['type'] == 'float':
                                # Generate random value with added noise for better diversity
                                range_size = range_info['max'] - range_info['min']
                                noise = random.gauss(0, range_size * 0.05)
                                rand_val = random.random() * range_size + range_info['min'] + noise
                                rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                                new_individual[param] = round(rand_val, 3)
                            elif range_info['type'] == 'categorical':
                                new_individual[param] = random.choice(range_info['values'])
                        else:
                            G = 2 * random.random() - 1
                            if range_info['type'] in ['int', 'float']:
                                term1 = a[param] + G * (a[param] - b[param])
                                term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                           ((c[param] - d[param]) - (e[param] - f[param])))
                                new_val = term1 + term2
                                new_val = max(range_info['min'], min(range_info['max'], new_val))
                                if range_info['type'] == 'int':
                                    new_individual[param] = int(round(new_val))
                                else:  # float
                                    new_individual[param] = round(new_val, 3)
                            elif range_info['type'] == 'categorical':
                                # For categorical, randomly choose from available values with some bias
                                if random.random() < 0.7:  # 70% chance to keep current value
                                    new_individual[param] = current[param]
                                else:
                                    new_individual[param] = random.choice(range_info['values'])
                
                # Check if this combination is unique
                param_tuple = tuple(new_individual.values())
                if param_tuple not in used_combinations:
                    used_combinations.add(param_tuple)
                    break
                attempts += 1
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Update pCR when no improvement (similar to MATLAB implementation)
                self.pCR = min(0.9, self.pCR + self.p)  # Cap at 0.9 to maintain some exploration
        
        return new_population, new_fitness
    
    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        
        # Convert to list of dictionaries with cost for easier manipulation
        Sol = [{'X': pop.copy(), 'Cost': fit} for pop, fit in zip(population, fitness_values)]
        NewSol = [{'X': {}, 'Cost': -np.inf} for _ in range(self.population_size)]
        
        # Get best solution
        best_idx = np.argmax(fitness_values)
        Best = {'X': population[best_idx].copy(), 'Cost': fitness_values[best_idx]}
        
        # Calculate mean position (mbest)
        mbest = {}
        for param in self.param_ranges.keys():
            range_info = self.param_ranges[param]
            if range_info['type'] in ['int', 'float']:
                mbest[param] = np.mean([s['X'][param] for s in Sol])
            elif range_info['type'] == 'categorical':
                # For categorical, find the most common value
                values = [s['X'][param] for s in Sol]
                from collections import Counter
                counter = Counter(values)
                mbest[param] = counter.most_common(1)[0][0]
        
        for i in range(self.population_size):
            # Generate random vectors
            beta1 = 2 * random.random()
            beta2 = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Generate w and v vectors (Eq 37, 38)
            w = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            v = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Calculate F1 and F2 (Eq 35, 36)
            F1 = {param: random.gauss(0, 1) * np.exp(2 - i * (2/self.generations)) 
                  for param in self.param_ranges.keys()}
            F2 = {param: w[param] * (v[param]**2) * np.cos((2 * random.random()) * w[param])
                  for param in self.param_ranges.keys()}
            
            # Calculate R_1 (Eq 34)
            R_1 = 2 * random.random() - 1
            
            # Calculate S1 and S2
            S1 = {param: (2 * random.random() - 1 + random.gauss(0, 1))
                  for param in self.param_ranges.keys()}
            S2 = {param: (F1[param] * R_1 * Sol[i]['X'][param] + 
                         F2[param] * (1 - R_1) * Best['X'][param])
                  for param in self.param_ranges.keys()}
            
            # Calculate VEC
            VEC = {param: S2[param] / S1[param] for param in self.param_ranges.keys()}
            
            if random.random() <= 0.5:
                Xatack = VEC
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(Sol)
                    for param in self.param_ranges.keys():
                        range_info = self.param_ranges[param]
                        if range_info['type'] in ['int', 'float']:
                            new_val = (Best['X'][param] + 
                                     beta1 * np.exp(beta2[param]) * 
                                     (random_sol['X'][param] - Sol[i]['X'][param]))
                            new_val = max(range_info['min'], min(range_info['max'], new_val))
                            if range_info['type'] == 'int':
                                NewSol[i]['X'][param] = int(round(new_val))
                            else:  # float
                                NewSol[i]['X'][param] = round(new_val, 3)
                        elif range_info['type'] == 'categorical':
                            # For categorical, use probabilistic selection
                            if random.random() < 0.5:
                                NewSol[i]['X'][param] = Best['X'][param]
                            else:
                                NewSol[i]['X'][param] = random_sol['X'][param]
                else:
                    # Eq 32 second part
                    for param in self.param_ranges.keys():
                        range_info = self.param_ranges[param]
                        if range_info['type'] in ['int', 'float']:
                            new_val = beta1 * Xatack[param] - Best['X'][param]
                            new_val = max(range_info['min'], min(range_info['max'], new_val))
                            if range_info['type'] == 'int':
                                NewSol[i]['X'][param] = int(round(new_val))
                            else:  # float
                                NewSol[i]['X'][param] = round(new_val, 3)
                        elif range_info['type'] == 'categorical':
                            # For categorical, probabilistic selection between attack vector and best
                            if random.random() < 0.7:
                                NewSol[i]['X'][param] = Best['X'][param]
                            else:
                                NewSol[i]['X'][param] = random.choice(range_info['values'])
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                sign = 1 if random.random() > 0.5 else -1
                for param in self.param_ranges.keys():
                    range_info = self.param_ranges[param]
                    if range_info['type'] in ['int', 'float']:
                        new_val = ((mbest[param] * Sol[r1]['X'][param] - 
                                  sign * Sol[i]['X'][param]) / 
                                 (1 + (Beta * random.random())))
                        new_val = max(range_info['min'], min(range_info['max'], new_val))
                        if range_info['type'] == 'int':
                            NewSol[i]['X'][param] = int(round(new_val))
                        else:  # float
                            NewSol[i]['X'][param] = round(new_val, 3)
                    elif range_info['type'] == 'categorical':
                        # For categorical, choose between mbest, random solution, or current
                        rand_choice = random.random()
                        if rand_choice < 0.4:
                            # Use most common value (approximating mbest for categorical)
                            values_count = {}
                            for s in Sol:
                                val = s['X'][param]
                                values_count[val] = values_count.get(val, 0) + 1
                            NewSol[i]['X'][param] = max(values_count.keys(), key=values_count.get)
                        elif rand_choice < 0.7:
                            NewSol[i]['X'][param] = Sol[r1]['X'][param]
                        else:
                            NewSol[i]['X'][param] = Sol[i]['X'][param]
            
            # Evaluate new solution
            NewSol[i]['Cost'] = self.evaluate_individual(NewSol[i]['X'])
            
            # Update solution (maximizing fitness)
            if NewSol[i]['Cost'] > Sol[i]['Cost']:
                Sol[i] = NewSol[i].copy()
        
        # Convert back to separate population and fitness arrays
        new_population = [s['X'] for s in Sol]
        new_fitness = [s['Cost'] for s in Sol]
        
        return new_population, new_fitness
    
    def optimize(self):
        """Run the PUMA optimization process"""
        # Initialize parameters
        UnSelected = [1, 1]  # [Exploration, Exploitation]
        F3_Explore = 0.001
        F3_Exploit = 0.001
        Seq_Time_Explore = [1.0, 1.0, 1.0]
        Seq_Time_Exploit = [1.0, 1.0, 1.0]
        Seq_Cost_Explore = [1.0, 1.0, 1.0]
        Seq_Cost_Exploit = [1.0, 1.0, 1.0]
        Score_Explore = 0.001
        Score_Exploit = 0.001
        PF = [0.5, 0.5, 0.3]  # Parameters for F1, F2, F3
        PF_F3 = []
        Mega_Explor = 0.99
        Mega_Exploit = 0.99
        Flag_Change = 1
        
        # Reset best metrics for new optimization run
        self.best_metrics = None
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Find initial best (since we're using negative RMSE, higher values are better)
        best_idx = np.argmax(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_best_fitness = best_fitness
        current_best_fitness = best_fitness
        
        print("\nOptimization Progress:")
        print("Gen |   R²   |  MAE   |  RMSE  | Fitness")
        print("-" * 45)
        
        # Track best scores for plotting
        self.best_scores_history = [best_fitness]

        # Unexperienced Phase (First 3 iterations)
        for Iter in range(3):
            # Exploration Phase
            pop_explor, fit_explor = self.exploration_phase(population, fitness_values)
            Costs_Explor = max(fit_explor)
            
            # Exploitation Phase
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            Costs_Exploit = max(fit_exploit)
            
            # Combine and sort solutions
            all_population = population + pop_explor + pop_exploit
            all_fitness = fitness_values + fit_explor + fit_exploit
            
            # Sort by fitness
            sorted_indices = np.argsort(all_fitness)[::-1]  # Descending order for maximization
            population = [all_population[i] for i in sorted_indices[:self.population_size]]
            fitness_values = [all_fitness[i] for i in sorted_indices[:self.population_size]]
            
            # Only update best if fitness improves
            if fitness_values[0] > current_best_fitness:
                best_individual = population[0].copy()
                best_fitness = fitness_values[0]
                current_best_fitness = best_fitness
            
            # Store best score for current iteration (keep previous best if no improvement)
            self.best_scores_history.append(current_best_fitness)
            
            # Print progress with simplified metrics
            latest_metrics = self.metrics_history[-1]['metrics']
            print(f"{Iter+1:3d} | {latest_metrics['r2']:6.4f} | {latest_metrics['mae']:6.4f} | "
                  f"{latest_metrics['rmse']:6.4f} | {current_best_fitness:6.4f}")
        
        # Calculate initial scores
        Seq_Cost_Explore[0] = abs(initial_best_fitness - Costs_Explor)
        Seq_Cost_Exploit[0] = abs(initial_best_fitness - Costs_Exploit)
        
        # Add non-zero costs to PF_F3
        for cost in Seq_Cost_Explore + Seq_Cost_Exploit:
            if cost != 0:
                PF_F3.append(cost)
        
        # Calculate initial F1 and F2 scores
        F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
        F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
        F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore))
        F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit))
        
        # Calculate initial scores
        Score_Explore = (PF[0] * F1_Explor) + (PF[1] * F2_Explor)
        Score_Exploit = (PF[0] * F1_Exploit) + (PF[1] * F2_Exploit)
        
        # Experienced Phase
        for Iter in range(3, self.generations):
            if Score_Explore > Score_Exploit:
                # Run Exploration
                SelectFlag = 1
                population, fitness_values = self.exploration_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[1] += 1
                UnSelected[0] = 1
                F3_Explore = PF[2]
                F3_Exploit += PF[2]
                
                # Update sequence costs for exploration
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
                Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
                Seq_Cost_Explore[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Explore[0] != 0:
                    PF_F3.append(Seq_Cost_Explore[0])
                
                if temp_best_fitness > current_best_fitness:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
            else:
                # Run Exploitation
                SelectFlag = 2
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[0] += 1
                UnSelected[1] = 1
                F3_Explore += PF[2]
                F3_Exploit = PF[2]
                
                # Update sequence costs for exploitation
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
                Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
                Seq_Cost_Exploit[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Exploit[0] != 0:
                    PF_F3.append(Seq_Cost_Exploit[0])
                
                if temp_best_fitness > current_best_fitness:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
            
            # Update time sequences if phase changed
            if Flag_Change != SelectFlag:
                Flag_Change = SelectFlag
                Seq_Time_Explore[2] = Seq_Time_Explore[1]
                Seq_Time_Explore[1] = Seq_Time_Explore[0]
                Seq_Time_Explore[0] = Count_select[0]
                Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
                Seq_Time_Exploit[1] = Seq_Time_Exploit[0]
                Seq_Time_Exploit[0] = Count_select[1]
            
            # Update F1 and F2 scores
            F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
            F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
            F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore))
            F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit))
            
            # Update Mega scores
            if Score_Explore < Score_Exploit:
                Mega_Explor = max((Mega_Explor - 0.01), 0.01)
                Mega_Exploit = 0.99
            elif Score_Explore > Score_Exploit:
                Mega_Explor = 0.99
                Mega_Exploit = max((Mega_Exploit - 0.01), 0.01)
            
            # Calculate lambda values
            lmn_Explore = 1 - Mega_Explor
            lmn_Exploit = 1 - Mega_Exploit
            
            # Update final scores
            Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (min(PF_F3) * F3_Explore))
            Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min(PF_F3) * F3_Exploit))
            
            # Store best score (keep previous best if no improvement)
            self.best_scores_history.append(current_best_fitness)
            
            # Print progress with simplified metrics
            latest_metrics = self.metrics_history[-1]['metrics']
            print(f"{Iter+1:3d} | {latest_metrics['r2']:6.4f} | {latest_metrics['mae']:6.4f} | "
                  f"{latest_metrics['rmse']:6.4f} | {current_best_fitness:6.4f}")
        
        # Store final results
        self.best_individual = best_individual
        self.best_score = best_fitness
        
        return self.best_individual, self.best_score

def plot_optimization_progress(scores_history):
    """Plot optimization progress - Modified for Kaggle environment"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores_history)
    plt.title('PUMA Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Best Composite Score')
    plt.grid(True)
    plt.savefig('puma_progress.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to save memory

def plot_feature_importance(model, feature_names):
    """Plot feature importance - Modified for Kaggle environment"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    
    # Cải thiện hiển thị nhãn trục x
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45,
               ha='right')
    
    # Thêm padding để tránh cắt nhãn
    plt.tight_layout(pad=2.0)
    
    # Thêm lưới để dễ đọc
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Thêm nhãn trục
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to save memory

def main():
    try:
        # Read CSV with semicolon separator - Kaggle input path
        df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
        
        # Feature columns for flood prediction
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Convert Yes/No to 1/0 for regression (probability of flood)
        df[label_column] = (df[label_column] == 'Yes').astype(float)
        
        # Replace comma with dot in numeric columns and convert to float
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Prepare data
        X = df[feature_columns].values
        y = np.array(df[label_column].values)
        
        # Handle missing values if any
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Initialize and run PUMA optimizer for RF with increased population size
        print("Starting PUMA optimization...")
        optimizer = PUMAOptimizer(X, y, population_size=25, generations=100)
        best_params, best_score = optimizer.optimize()
        
        # Plot optimization progress - Save but don't show for Kaggle
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['r2', 'mae', 'rmse']  # Simplified to 3 metrics
        colors = ['b', 'g', 'r']
        
        for metric, color in zip(metrics_to_plot, colors):
            # Get metric values for each generation
            metric_values = []
            for gen_idx in range(optimizer.generations):
                gen_metrics = [m['metrics'][metric] for m in optimizer.metrics_history 
                             if m['generation'] == gen_idx]
                if gen_metrics:
                    metric_values.append(gen_metrics[-1])
            
            # Plot with appropriate scaling for different metrics
            if metric in ['mae', 'rmse']:
                # Use log scale for error metrics
                plt.semilogy(range(len(metric_values)), metric_values, f'{color}-', label=metric.upper())
            else:
                plt.plot(range(len(metric_values)), metric_values, f'{color}-', label=metric.upper())
        
        plt.title('PUMA Optimization Progress - Simplified Metrics (R², MAE, RMSE)')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('po_rf_optimization_progress.png', dpi=150, bbox_inches='tight')
        # Remove plt.show() for Kaggle environment
        plt.close()  # Close figure to save memory
        
        # Print final results
        print("\n=== Final Results ===")
        print(f"Best RMSE Score: {-best_score:.6f}")  # Convert back from negative
        print("\nOptimal Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        # Train final model with best parameters using appropriate implementation
        if optimizer.use_gpu and GPU_AVAILABLE:
            final_model = cuRandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=RANDOM_SEED
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            final_model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                bootstrap=best_params['bootstrap'],
                max_samples=best_params['max_samples'] if best_params['bootstrap'] else None,
                n_jobs=-1,
                random_state=RANDOM_SEED
            )
        
        # Train and evaluate on test set
        final_model.fit(optimizer.X_train_scaled, optimizer.y_train)
        y_pred = final_model.predict(optimizer.X_test_scaled)
        
        # Handle GPU arrays conversion if needed
        if optimizer.use_gpu and GPU_AVAILABLE:
            if hasattr(y_pred, 'get'):
                y_pred = y_pred.get()
            if hasattr(optimizer.y_test, 'get'):
                y_test = optimizer.y_test.get()
            else:
                y_test = np.array(optimizer.y_test)
        else:
            y_pred = np.array(y_pred)
            y_test = np.array(optimizer.y_test)
            
        y_pred = np.clip(y_pred, 0, 1)  # Clip predictions between 0 and 1
        
        # Calculate final metrics using sklearn
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        final_r2 = r2_score(y_test, y_pred)
        final_mae = mean_absolute_error(y_test, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Create simplified results DataFrame with only 3 key metrics
        results_data = {
            # Final Test Metrics (3 key metrics only)
            'Final_R2': [final_r2],
            'Final_MAE': [final_mae],
            'Final_RMSE': [final_rmse],
            
            # Best Parameters from Optimization
            'Best_n_estimators': [best_params['n_estimators']],
            'Best_max_depth': [best_params['max_depth']],
            'Best_min_samples_split': [best_params['min_samples_split']],
            'Best_min_samples_leaf': [best_params['min_samples_leaf']],
            'Best_max_features': [best_params['max_features']],
            'Best_bootstrap': [best_params['bootstrap']],
            'Best_max_samples': [best_params['max_samples']],
            
            # Optimization Details
            'Best_RMSE_Score': [-best_score],  # Convert back from negative
            'Total_Generations': [optimizer.generations],
            'Population_Size': [optimizer.population_size],
            'Random_Seed': [RANDOM_SEED],
            'GPU_Used': [optimizer.use_gpu and GPU_AVAILABLE],
            
            # Dataset Information
            'Train_Samples': [len(optimizer.X_train_scaled)],
            'Test_Samples': [len(optimizer.X_test_scaled)],
            'Feature_Count': [len(feature_columns)]
        }
        
        # Add best metrics from optimization history (3 metrics only)
        if optimizer.best_metrics:
            results_data.update({
                'Opt_Best_R2': [optimizer.best_metrics['r2']],
                'Opt_Best_MAE': [optimizer.best_metrics['mae']],
                'Opt_Best_RMSE': [optimizer.best_metrics['rmse']]
            })
        
        # Convert to DataFrame and save
        comprehensive_results = pd.DataFrame(results_data)
        comprehensive_results.to_csv('po_rf_comprehensive_results.csv', index=False)
        
        # Print summary for Kaggle logs
        print("\n=== PUMA-RF Optimization Complete ===")
        print(f"Best RMSE Score: {final_rmse:.6f}")
        print(f"Final Test R²: {final_r2:.6f}")
        print(f"Final Test MAE: {final_mae:.6f}")
        print(f"GPU Used: {optimizer.use_gpu and GPU_AVAILABLE}")
        print(f"Results saved to: po_rf_comprehensive_results.csv")
        print(f"Optimization plot saved to: po_rf_optimization_progress.png")
            
    except FileNotFoundError:
        print("File not found! Please check the dataset path.")
        print("Expected path: /kaggle/input/data-rf-po/flood_training.csv")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()