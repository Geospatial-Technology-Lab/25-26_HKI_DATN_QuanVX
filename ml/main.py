import numpy as np
import os
from datetime import datetime
from optimization.rs_optimizer import RandomizedSearch
from optimization.pso_optimizer import create_optimizer as create_pso_optimizer
from optimization.puma_optimizer import PUMAOptimizer
from evaluation.evaluation_utils import load_data_from_csv

# Tạo thư mục results nếu chưa có
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load dữ liệu thực
data_path = r"D:\25-26_HKI_DATN_QuanVX\train\data\training_data.csv"
X_train, X_test, y_train, y_test = load_data_from_csv(data_path, test_size=0.3, random_state=42)
X, y = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])

optimizers = ['RS', 'PSO', 'PUMA']
models = ['rf', 'xgb', 'svm']

for opt in optimizers:
    for model in models:
        print(f"\n{'='*80}")
        print(f"  {opt} - {model.upper()}")
        print(f"{'='*80}")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if opt == 'RS':
                rs = RandomizedSearch(X, y, model, random_state=42)
                result = rs.search(n_iter=100, verbose=True)
                
                # Lưu kết quả ra CSV
                csv_filename = os.path.join(results_dir, f"randomized_search_{model}_results_{timestamp}.csv")
                rs.save_results_to_csv(csv_filename)
                print(f"\n✓ Đã lưu kết quả vào: {csv_filename}")
                
            elif opt == 'PSO':
                pso = create_pso_optimizer(model, X, y, n_particles=10, n_iterations=50, random_state=42)
                result = pso.optimize(verbose=True)
                
                # Lưu kết quả ra CSV
                csv_filename = os.path.join(results_dir, f"pso_{model}_optimization_results_{timestamp}.csv")
                pso.save_results_to_csv(csv_filename)
                print(f"\n✓ Đã lưu kết quả vào: {csv_filename}")
                
            elif opt == 'PUMA':
                puma = PUMAOptimizer(X, y, population_size=10, generations=100, random_state=42)
                puma.set_param_ranges_by_model_type(model)
                result = puma.optimize(verbose=True)
                
                # Lưu kết quả ra CSV
                csv_filename = os.path.join(results_dir, f"puma_{model}_optimization_results_{timestamp}.csv")
                puma.save_results_to_csv(csv_filename)
                print(f"\n✓ Đã lưu kết quả vào: {csv_filename}")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
