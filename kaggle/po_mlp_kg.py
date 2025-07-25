import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
import warnings
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings('ignore')

# Removed GPU-related code

class PUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        # Prepare data for processing
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data(X, y)
        
        # Initialize optimization parameters
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = np.inf
        self.best_scores_history = []
        self.metrics_history = []
        self.best_metrics = None
        self.pCR = 0.5
        self.p = 0.1
        
        # Initialize evaluation counter for better tracking
        self.evaluation_count = 0
        
        # Track monotonic improvement (ensure fitness only improves)
        self.best_rmse_ever = np.inf  # Track best RMSE (lower is better)
        self.monotonic_fitness_history = []  # Only record improvements
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Parameter search space for MLP regression
        self.param_ranges = {
            'hidden_layer_sizes': {
                'type': 'choice',
                'choices': [
                    (32,), (64,), (100,), (128,), (256,), (512,),
                    (32, 16), (64, 32), (100, 50), (128, 64), (256, 128),
                    (100, 100), (128, 128), (256, 256),
                    (100, 50, 25), (128, 64, 32), (256, 128, 64)
                ]
            },
            'activation': {
                'type': 'choice', 
                'choices': ['relu', 'tanh', 'logistic']
            },
            'solver': {
                'type': 'choice',
                'choices': ['adam', 'lbfgs']
            },
            'alpha': {'type': 'float', 'min': 1e-6, 'max': 1e-2},
            'learning_rate': {
                'type': 'choice',
                'choices': ['constant', 'adaptive', 'invscaling']
            },
            'learning_rate_init': {'type': 'float', 'min': 1e-4, 'max': 1e-2},
            'max_iter': {'type': 'int', 'min': 500, 'max': 2000},
            'beta_1': {'type': 'float', 'min': 0.85, 'max': 0.95},
            'beta_2': {'type': 'float', 'min': 0.9, 'max': 0.999},
            'epsilon': {'type': 'float', 'min': 1e-9, 'max': 1e-7}
        }
    
    def _prepare_data(self, X, y):
        """Xử lý và chuẩn bị dữ liệu"""
        # Convert to numpy first to handle missing values
        if isinstance(X, pd.DataFrame):
            X_filled = X.fillna(X.mean())
            X_np = X_filled.values
        else:
            X_np = np.nan_to_num(X, nan=0.0)
            
        if isinstance(y, pd.Series):
            y_filled = y.fillna(y.mean())
            y_np = y_filled.values
        else:
            y_np = np.nan_to_num(y, nan=0.0)
        
        # Clean any remaining NaN/inf
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=3.0, neginf=-3.0)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Split data - check if stratification is possible
        unique_values = np.unique(y_np)
        if len(unique_values) > 1 and np.min(np.bincount(y_np.astype(int))) >= 2:
            # Use stratification if possible (without fixed random state)
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, stratify=y_np
            )
        else:
            # Use regular split if stratification not possible (without fixed random state)
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2
            )
        
        # Return the split data directly
        return X_train, X_test, y_train, y_test
    
    def _to_numpy(self, data):
        """Chuyển đổi dữ liệu về numpy format"""
        if hasattr(data, 'values'):  # pandas objects
            return data.values
        return np.array(data)  # fallback
    
    def create_individual(self):
        """Create a random individual (parameter set)"""
        # Create individual with parameter values based on type
        temp_individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'choice':
                temp_individual[param] = random.choice(range_info['choices'])
            else:  # int or float
                range_size = range_info['max'] - range_info['min']
                noise_factor = 0.1 if range_info['type'] == 'int' else 0.05
                noise = random.gauss(0, range_size * noise_factor)
                value = random.random() * range_size + range_info['min'] + noise
                # Apply constraints and format based on type
                value = max(range_info['min'], min(range_info['max'], value))
                # Special case for epsilon to ensure it's never zero
                if param == 'epsilon' and value <= 0:
                    value = range_info['min']  # Use minimum allowed value
                temp_individual[param] = int(round(value)) if range_info['type'] == 'int' else round(value, 6)
        
        return temp_individual
    
    def evaluate_individual(self, individual):
        try:
            self.evaluation_count += 1
            
            # Make a safe copy of individual params and ensure epsilon > 0
            safe_individual = individual.copy()
            if 'epsilon' in safe_individual and safe_individual['epsilon'] <= 0:
                safe_individual['epsilon'] = self.param_ranges['epsilon']['min']
            
            # Limit max_iter to prevent excessive training time
            if 'max_iter' in safe_individual and safe_individual['max_iter'] > 1000:
                safe_individual['max_iter'] = 1000  # Cap at 1000
            
            # Validate parameters for NaN/inf
            for param_name, param_value in safe_individual.items():
                # Handle tuple parameters (like hidden_layer_sizes)
                if isinstance(param_value, tuple):
                    if any(np.isnan(v) or np.isinf(v) for v in param_value):
                        return -np.inf
                elif isinstance(param_value, (int, float)):
                    if np.isnan(param_value) or np.isinf(param_value):
                        return -np.inf
            
            # CPU implementation with optimizations for speed
            model = MLPRegressor(
                **safe_individual,
                tol=1e-3,  # Less strict tolerance for faster convergence
                early_stopping=True,
                validation_fraction=0.15,  # Larger validation set
                n_iter_no_change=5,  # Stop earlier if no improvement
                warm_start=False
            )
            
            # Prepare data for training
            X_train_np = self._to_numpy(self.X_train_scaled)
            y_train_np = self._to_numpy(self.y_train)
            X_test_np = self._to_numpy(self.X_test_scaled)
            y_test_np = self._to_numpy(self.y_test)
            
            # Train model and get predictions
            model.fit(X_train_np, y_train_np)
            y_pred_np = model.predict(X_test_np)
            
            # Clip predictions for binary classification
            y_pred_np = np.clip(y_pred_np, 0, 1)
            
            # Calculate metrics
            r2 = r2_score(y_test_np, y_pred_np)
            mae = mean_absolute_error(y_test_np, y_pred_np)
            rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
            
            # Store simplified metrics
            current_metrics = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
            
            # Initialize best_metrics if not set (using RMSE for comparison - lower is better)
            if self.best_metrics is None:
                self.best_metrics = current_metrics.copy()
                self.best_rmse_ever = rmse
                # Only store essential metrics
                self.metrics_history.append({
                    'generation': len(self.metrics_history),
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse
                })
            else:
                # Update ONLY if RMSE improves (decreases)
                if rmse < self.best_rmse_ever:
                    self.best_metrics = current_metrics.copy()
                    self.best_rmse_ever = rmse
                
                # Only store essential metrics
                self.metrics_history.append({
                    'generation': len(self.metrics_history),
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse
                })
            
            # Return RMSE as fitness (lower is better, so we'll negate it for maximization)
            return float(-rmse)  # Negative because optimization algorithms typically maximize
            
        except Exception as e:
            # Return very poor fitness for failed evaluations
            return -np.inf
            
    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions randomly
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Create new solution
            new_individual = current.copy()  # Start with current solution
            
            # Ensure at least one parameter changes by selecting random parameter
            j0 = random.choice(list(self.param_ranges.keys()))
            
            # Generate random value with added noise for better diversity
            for param, range_info in self.param_ranges.items():
                # Always change j0 parameter or based on pCR probability
                if param == j0 or random.random() <= self.pCR:
                    if random.random() < 0.5:
                        if range_info['type'] == 'int':
                            # Generate random value with added noise for better diversity
                            range_size = range_info['max'] - range_info['min']
                            noise = random.gauss(0, range_size * 0.1)
                            rand_val = random.random() * range_size + range_info['min'] + noise
                            new_individual[param] = int(round(max(range_info['min'], min(range_info['max'], rand_val))))
                        elif range_info['type'] == 'float':
                            # Generate random value with added noise for better diversity
                            range_size = range_info['max'] - range_info['min']
                            noise = random.gauss(0, range_size * 0.05)
                            rand_val = random.random() * range_size + range_info['min'] + noise
                            new_individual[param] = round(max(range_info['min'], min(range_info['max'], rand_val)), 6)
                        elif range_info['type'] == 'choice':
                            new_individual[param] = random.choice(range_info['choices'])
                    else:
                        G = 2 * random.random() - 1
                        if range_info['type'] in ['int', 'float']:
                            term1 = a[param] + G * (a[param] - b[param])
                            term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                       ((c[param] - d[param]) - (e[param] - f[param])))
                            new_val = term1 + term2
                            if range_info['type'] == 'int':
                                new_individual[param] = int(round(max(range_info['min'], min(range_info['max'], new_val))))
                            else:  # float
                                value = max(range_info['min'], min(range_info['max'], new_val))
                                # Special case for epsilon to ensure it's never zero
                                if param == 'epsilon' and value <= 0:
                                    value = range_info['min']  # Use minimum allowed value
                                new_individual[param] = round(value, 6)
                        elif range_info['type'] == 'choice':
                            # For choice, randomly choose from available values with some bias
                            if random.random() < 0.7:  # 70% chance to keep current value
                                new_individual[param] = current[param]
                            else:
                                new_individual[param] = random.choice(range_info['choices'])
            
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
            elif range_info['type'] == 'choice':
                # For choice, find the most common value
                values = [s['X'][param] for s in Sol]
                from collections import Counter
                counter = Counter(str(v) for v in values)  # Convert to string for hashing
                most_common_str = counter.most_common(1)[0][0]
                # Convert back to original type
                for s in Sol:
                    if str(s['X'][param]) == most_common_str:
                        mbest[param] = s['X'][param]
                        break
        
        for i in range(self.population_size):
            # Generate random vectors
            beta1 = 2 * random.random()
            beta2 = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Generate w và v vectors (Eq 37, 38)
            w = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            v = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Calculate F1, F2, R_1, S1, S2 và VEC
            R_1 = 2 * random.random() - 1
            params = self.param_ranges.keys()
            
            F1 = {p: random.gauss(0, 1) * np.exp(2 - i * (2/self.generations)) for p in params}
            F2 = {p: w[p] * (v[p]**2) * np.cos((2 * random.random()) * w[p]) for p in params}
            S1 = {p: (2 * random.random() - 1 + random.gauss(0, 1)) for p in params}
            
            # Handle special case for parameters that could be tuples (like hidden_layer_sizes)
            S2 = {}
            for p in params:
                range_info = self.param_ranges[p]
                if range_info['type'] == 'choice':
                    # For choice parameters, don't do mathematical operations
                    S2[p] = Sol[i]['X'][p]
                else:
                    # For numeric parameters, perform the calculation
                    S2[p] = F1[p] * R_1 * Sol[i]['X'][p] + F2[p] * (1 - R_1) * Best['X'][p]
                    
            # Calculate VEC with type checking
            VEC = {}
            for p in params:
                range_info = self.param_ranges[p]
                if range_info['type'] == 'choice':
                    VEC[p] = S2[p]  # For choice parameters, just use the value directly
                else:
                    VEC[p] = S2[p] / S1[p]  # For numeric parameters, do the division
            
            if random.random() <= 0.5:
                Xatack = VEC
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(Sol)
                    for param in self.param_ranges.keys():
                        range_info = self.param_ranges[param]
                        if range_info['type'] == 'choice':
                            # For choice parameters, use probabilistic selection
                            if random.random() < 0.5:
                                NewSol[i]['X'][param] = Best['X'][param]
                            else:
                                NewSol[i]['X'][param] = random_sol['X'][param]
                        elif range_info['type'] in ['int', 'float']:
                            # For numeric parameters, perform mathematical operations
                            new_val = (Best['X'][param] + 
                                     beta1 * np.exp(beta2[param]) * 
                                     (random_sol['X'][param] - Sol[i]['X'][param]))
                            if range_info['type'] == 'int':
                                NewSol[i]['X'][param] = int(round(max(range_info['min'], min(range_info['max'], new_val))))
                            else:  # float
                                NewSol[i]['X'][param] = round(max(range_info['min'], min(range_info['max'], new_val)), 6)
                else:
                    # Eq 32 second part
                    for param in self.param_ranges.keys():
                        range_info = self.param_ranges[param]
                        if range_info['type'] == 'choice':
                            # For choice parameters, probabilistic selection
                            if random.random() < 0.7:
                                NewSol[i]['X'][param] = Best['X'][param]
                            else:
                                NewSol[i]['X'][param] = random.choice(range_info['choices'])
                        elif range_info['type'] in ['int', 'float']:
                            # For numeric parameters, perform mathematical operations
                            new_val = beta1 * Xatack[param] - Best['X'][param]
                            if range_info['type'] == 'int':
                                NewSol[i]['X'][param] = int(round(max(range_info['min'], min(range_info['max'], new_val))))
                            else:  # float
                                NewSol[i]['X'][param] = round(max(range_info['min'], min(range_info['max'], new_val)), 6)
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                sign = 1 if random.random() > 0.5 else -1
                for param in self.param_ranges.keys():
                    range_info = self.param_ranges[param]
                    if range_info['type'] == 'choice':
                        # For choice parameters, probabilistic selection between options
                        rand_choice = random.random()
                        if rand_choice < 0.4:
                            NewSol[i]['X'][param] = mbest[param]
                        elif rand_choice < 0.7:
                            NewSol[i]['X'][param] = Sol[r1]['X'][param]
                        else:
                            NewSol[i]['X'][param] = Sol[i]['X'][param]
                    elif range_info['type'] in ['int', 'float']:
                        # For numeric parameters, perform mathematical operations
                        new_val = ((mbest[param] * Sol[r1]['X'][param] - 
                                  sign * Sol[i]['X'][param]) / 
                                 (1 + (Beta * random.random())))
                        if range_info['type'] == 'int':
                            NewSol[i]['X'][param] = int(round(max(range_info['min'], min(range_info['max'], new_val))))
                        else:  # float
                            value = max(range_info['min'], min(range_info['max'], new_val))
                            # Special case for epsilon
                            if param == 'epsilon' and value <= 0:
                                value = range_info['min']
                            NewSol[i]['X'][param] = round(value, 6)
            
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
        """Run the PUMA optimization process - Keeping original PUMA logic"""
        # Khởi tạo các tham số thuật toán PUMA gốc
        UnSelected = [1, 1]  # [Exploration, Exploitation]
        F3_Explore = F3_Exploit = 0.001
        Seq_Time_Explore = Seq_Time_Exploit = [1.0, 1.0, 1.0]
        Seq_Cost_Explore = Seq_Cost_Exploit = [1.0, 1.0, 1.0]
        Score_Explore = Score_Exploit = 0.001
        PF = [0.5, 0.5, 0.3]  # Parameters for F1, F2, F3
        PF_F3 = []
        Mega_Explor = Mega_Exploit = 0.99
        Flag_Change = 1
        
        # Reset best metrics for new optimization run
        self.best_metrics = None
        self.best_rmse_ever = np.inf
        self.monotonic_fitness_history = []
        
        # Initialize population
        population = []
        fitness_values = []
        
        # Create population with error handling
        for i in range(self.population_size):
            individual = self.create_individual()
            fitness = self.evaluate_individual(individual)
            
            # Ensure we have valid fitness
            if fitness == -np.inf:
                # Try a few more times with different random parameters
                for attempt in range(3):
                    individual = self.create_individual()
                    fitness = self.evaluate_individual(individual)
                    if fitness != -np.inf:
                        break
                
                # If still failed, use a very basic individual optimized for speed
                if fitness == -np.inf:
                    individual = {
                        'hidden_layer_sizes': (100,),  # Standard network
                        'activation': 'relu',
                        'solver': 'adam',
                        'alpha': 0.001,
                        'learning_rate': 'constant',
                        'learning_rate_init': 0.001,
                        'max_iter': 500,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'epsilon': 1e-8
                    }
                    fitness = self.evaluate_individual(individual)
            
            population.append(individual)
            fitness_values.append(fitness)
        
        # Find initial best (since we're using negative RMSE, higher values are better)
        best_idx = np.argmax(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_best_fitness = best_fitness
        current_best_fitness = best_fitness
        
        # Initialize monotonic tracking
        current_best_rmse = -best_fitness  # Convert back to positive RMSE
        current_best_mae = self.best_metrics['mae'] if self.best_metrics else np.inf
        current_best_r2 = self.best_metrics['r2'] if self.best_metrics else -np.inf
        self.monotonic_fitness_history.append(current_best_rmse)
        
        print(f"\nStarting optimization with initial metrics:")
        print(f"{'Generation':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12}")
        print("-" * 50)
        print(f"{'0':<12} {current_best_rmse:<12.6f} {current_best_mae:<12.6f} {current_best_r2:<12.6f}")
        
        # Track best scores for plotting
        self.best_scores_history = [best_fitness]

        # Unexperienced Phase (First 3 iterations) - PUMA gốc
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
            
            # Only update best if fitness improves (RMSE decreases)
            current_rmse = -fitness_values[0]  # Convert back to positive RMSE
            if current_rmse < current_best_rmse:
                best_individual = population[0].copy()
                best_fitness = fitness_values[0]
                current_best_fitness = best_fitness
                current_best_rmse = current_rmse
                current_best_mae = self.best_metrics['mae']
                current_best_r2 = self.best_metrics['r2']
                self.monotonic_fitness_history.append(current_best_rmse)
            else:
                # Keep the same best RMSE in monotonic history
                self.monotonic_fitness_history.append(current_best_rmse)
            
            # Print metrics for current generation
            print(f"{Iter+1:<12} {current_best_rmse:<12.6f} {current_best_mae:<12.6f} {current_best_r2:<12.6f}")
            
            # Store best score for current iteration (keep previous best if no improvement)
            self.best_scores_history.append(current_best_fitness)
        
        # Calculate initial scores - PUMA gốc
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
        
        # Experienced Phase - PUMA gốc
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
                phase_name = "Exploration"
                
                # Update sequence costs for exploration
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
                Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
                Seq_Cost_Explore[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Explore[0] != 0:
                    PF_F3.append(Seq_Cost_Explore[0])
                
                # Update best only if RMSE improves
                temp_rmse = -temp_best_fitness
                if temp_rmse < current_best_rmse:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
                    current_best_rmse = temp_rmse
                    current_best_mae = self.best_metrics['mae']
                    current_best_r2 = self.best_metrics['r2']
                    self.monotonic_fitness_history.append(current_best_rmse)
                else:
                    self.monotonic_fitness_history.append(current_best_rmse)
                
                # Print metrics for current generation
                print(f"{Iter+1:<12} {current_best_rmse:<12.6f} {current_best_mae:<12.6f} {current_best_r2:<12.6f}")
            else:
                # Run Exploitation
                SelectFlag = 2
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[0] += 1
                UnSelected[1] = 1
                F3_Explore += PF[2]
                F3_Exploit = PF[2]
                phase_name = "Exploitation"
                
                # Update sequence costs for exploitation
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
                Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
                Seq_Cost_Exploit[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Exploit[0] != 0:
                    PF_F3.append(Seq_Cost_Exploit[0])
                
                # Update best only if RMSE improves
                temp_rmse = -temp_best_fitness
                if temp_rmse < current_best_rmse:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
                    current_best_rmse = temp_rmse
                    current_best_mae = self.best_metrics['mae']
                    current_best_r2 = self.best_metrics['r2']
                    self.monotonic_fitness_history.append(current_best_rmse)
                else:
                    self.monotonic_fitness_history.append(current_best_rmse)
                
                # Print metrics for current generation
                print(f"{Iter+1:<12} {current_best_rmse:<12.6f} {current_best_mae:<12.6f} {current_best_r2:<12.6f}")
            
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
            if len(PF_F3) > 0:
                Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (min(PF_F3) * F3_Explore))
                Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min(PF_F3) * F3_Exploit))
            else:
                Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor)
                Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit)
            
            # Store best score (keep previous best if no improvement)
            self.best_scores_history.append(current_best_fitness)
        
        # Store final results
        self.best_individual = best_individual
        self.best_score = best_fitness
        
        return self.best_individual, self.best_score

def main():
    try:
        # Data preparation - all in one block with clearer comments
        feature_columns = ['Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road', 
                          'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
                          'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall']
        label_column = 'Nom'
        
        # Load and preprocess data in fewer steps
        df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
        df[label_column] = (df[label_column] == 'Yes').astype(float)  # Convert Yes/No to 1/0
        
        # Fix number format and extract features in one pass
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
                
        # Create feature matrix and target vector
        X = df[feature_columns].values
        y = np.array(df[label_column].values)
        
        # Handle missing values if needed
        if pd.isna(X).any().any():
            from sklearn.impute import SimpleImputer
            X = SimpleImputer(strategy='median').fit_transform(X)
        
        # Initialize and run PUMA optimizer for MLP
        print("Starting PUMA optimization for MLP...")
        
        # Use standard population and generations
        optimizer = PUMAOptimizer(X, y, population_size=10, generations=100)
        
        best_params, best_score = optimizer.optimize()
        
        # Create optimization progress plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Fitness progression (allows increases)
        ax1.plot(range(len(optimizer.best_scores_history)), optimizer.best_scores_history, 'b-', label='Fitness Score')
        ax1.set_title('PUMA Fitness Progression')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness Score')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Monotonic RMSE improvement (only decreases)
        ax2.plot(range(len(optimizer.monotonic_fitness_history)), optimizer.monotonic_fitness_history, 'r-', label='Best RMSE (Monotonic)')
        ax2.set_title('PUMA RMSE Improvement (Monotonic)')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('RMSE (Lower is Better)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('po_mlp_optimization_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print final results
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETED")
        print("="*50)
        print(f"Final RMSE: {optimizer.best_rmse_ever:.6f}")
        print(f"Final MAE:  {optimizer.best_metrics['mae']:.6f}")
        print(f"Final R2:   {optimizer.best_metrics['r2']:.6f}")
        print(f"Total Evaluations: {optimizer.evaluation_count}")
        improvements = len([x for i, x in enumerate(optimizer.monotonic_fitness_history[1:], 1) if x < optimizer.monotonic_fitness_history[i-1]])
        print(f"RMSE Improvements: {improvements}")
        
        print("\nOptimal Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print("\nOptimization completed successfully!")
        

    except FileNotFoundError:
        print("File not found! Please check the dataset path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()