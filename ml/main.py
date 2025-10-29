import numpy as np
from optimization.rso_optimizer import RandomizedSearch
from optimization.pso_optimizer import create_optimizer as create_pso_optimizer
from optimization.puma_optimizer import PUMAOptimizer
from evaluation.evaluation_utils import load_data_from_csv

# Load dữ liệu thực
data_path = r"D:\Vscode\gee\flood_points_3k_normalized.csv"
X_train, X_test, y_train, y_test = load_data_from_csv(data_path, test_size=0.3, random_state=42)
X, y = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])

optimizers = ['RSO', 'PSO', 'PUMA']
models = ['rf', 'xgb', 'svm']

for opt in optimizers:
    for model in models:
        print(f"\n=== {opt} - {model.upper()} ===")
        
        try:
            if opt == 'RSO':
                rs = RandomizedSearch(X, y, model, random_state=42)
                rs.search(n_iter=100, verbose=True)
                
            elif opt == 'PSO':
                pso = create_pso_optimizer(model, X, y, n_particles=10, n_iterations=100, random_state=42)
                pso.optimize(verbose=True)
                
            elif opt == 'PUMA':
                puma = PUMAOptimizer(X, y, population_size=10, generations=100, random_state=42)
                puma.set_param_ranges_by_model_type(model)
                puma.optimize(verbose=True)
                
        except Exception as e:
            print(f"Error: {str(e)}")
