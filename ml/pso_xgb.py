"""XGBoost PSO Optimizer - Tối ưu hóa XGBoost bằng PSO (Phiên bản tối giản)."""

import time
from sklearn.model_selection import train_test_split

# Import tất cả từ các module có sẵn
from pso_optimizer import XGBoostPSOOptimizer, RANDOM_SEED
from data_preprocessing import prepare_flood_data, get_feature_info
from model_params import OPTIMIZATION_CONFIG
from evaluation_utils import evaluate_regression_model


def main() -> None:
    """Hàm chính để chạy quá trình tối ưu hóa XGBoost."""
    try:
        # === CHUẨN BỊ DỮ LIỆU ===
        print("Bắt đầu chuẩn bị dữ liệu...")
        X, y, feature_columns = prepare_flood_data(shuffle_data=True, debug=True)
        feature_names, label_column = get_feature_info()
        
        # === HIỂN THỊ CẤU HÌNH ===
        n_particles = OPTIMIZATION_CONFIG.get('population_size')
        n_iterations = OPTIMIZATION_CONFIG.get('generations')
        
        print(f"\nCấu hình PSO:")
        print(f"- Số hạt (n_particles): {n_particles}")
        print(f"- Số vòng lặp (n_iterations): {n_iterations}")
        print(f"- Random seed: {RANDOM_SEED}")
        print(f"- Số lượng đặc trưng: {len(feature_names)}")
        
        # === CHẠY TỐI ƯU HÓA ===
        xgb_optimizer = XGBoostPSOOptimizer(X, y, 'regression', n_particles, n_iterations)
        
        start_time = time.time()
        result = xgb_optimizer.optimize(verbose=True, print_table=True, save_csv=True, save_model=True)
        end_time = time.time()
        
        print(f"\nThời gian tối ưu hóa: {end_time - start_time:.2f} giây")
        
        # === ĐÁNH GIÁ MODEL CUỐI CÙNG ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Đánh giá cuối cùng
        final_metrics = evaluate_regression_model(
            result['best_model'], X_train, X_test,
            y_train, y_test, return_detailed=True
        )
        
        # === HIỂN THỊ KẾT QUẢ ===
        print(f"\n{'='*60}")
        print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST")
        print(f"{'='*60}")
        print(f"R²: {final_metrics['r2']:.4f}")
        print(f"RMSE: {final_metrics['rmse']:.4f}")
        print(f"MAE: {final_metrics['mae']:.4f}")
        print(f"Fitness: {final_metrics['fitness']:.4f}")
        
        print(f"\n{'='*60}")
        print("ĐÃ LƯU KẾT QUẢ")
        print(f"{'='*60}")
        if result.get('csv_file'):
            print(f"Results CSV: {result['csv_file']}")
        if result.get('model_file'):
            print(f"Best Model: {result['model_file']}")
        
        print(f"\n{'='*60}")
        print("HOÀN THÀNH TỐI ƯU HÓA XGBOOST")
        print(f"{'='*60}")
        
        return result['best_model'], result['best_params'], final_metrics
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file dữ liệu! {e}")
        print("Vui lòng kiểm tra đường dẫn dataset trong config.py")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()