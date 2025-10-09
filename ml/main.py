import numpy as np
from optimization.pso_optimizer import quick_optimize as pso_optimize
from optimization.rso_optimizer import quick_randomized_search
from optimization.puma_optimizer import PUMAOptimizer
from evaluation.evaluation_utils import load_data_from_csv

# Load dữ liệu thực
data_path = r"C:\Users\Admin\Downloads\Compressed\25-26_HKI_DATN_QuanVX-main\25-26_HKI_DATN_QuanVX-main\train\data\training_points.csv"
X_train, X_test, y_train, y_test = load_data_from_csv(data_path, test_size=0.2, random_state=42)
X, y = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])

models = ['rf', 'xgb', 'svm', 'mlp']

for model in models:
    print(f"\n=== {model.upper()} ===")
    
    # Randomized Search
    try:
        rs = quick_randomized_search(model, X, y, n_iter=100, random_state=42)
        print(f"RS: {rs['best_score']:.4f}")
    except: print("RS: Error")
    
    # PSO
    try:
        pso = pso_optimize(model, X, y, n_particles=10, n_iterations=100, random_state=42)
        print(f"PSO: {pso['best_score']:.4f}")
    except: print("PSO: Error")
    
    # PUMA
    try:
        puma = PUMAOptimizer(X, y, population_size=10, generations=100, random_state=42)
        puma.set_param_ranges_by_model_type(model)
        result = puma.optimize(verbose=False)
        print(f"PUMA: {result['best_score']:.4f}")
    except: print("PUMA: Error")