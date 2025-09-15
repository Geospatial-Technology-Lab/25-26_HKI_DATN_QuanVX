# Machine Learning Optimization Framework for Flood Prediction

## Tổng quan
Dự án này cung cấp một framework tối ưu hóa tham số cho các mô hình machine learning nhằm dự đoán lũ lụt. Framework hỗ trợ nhiều thuật toán tối ưu hóa khác nhau (PSO, RSO, PUMA) cho các mô hình machine learning phổ biến (Random Forest, SVM, XGBoost, MLP).

## 🔄 **Cập nhật gần đây**
- ✅ **Cải thiện MLP Optimization**: Thêm hỗ trợ đầy đủ cho 13 tham số MLP
- ✅ **Sửa lỗi PUMA cho MLP**: Tạo hàm `calculate_mlp_metrics()` riêng biệt để đảm bảo metrics hiển thị chính xác
- ✅ **Loại bỏ hiển thị Phase**: Đơn giản hóa bảng kết quả, chỉ hiển thị metrics quan trọng
- ✅ **Tổ chức lại config**: Tách biệt config general và model-specific parameters
- ✅ **Cải thiện stability**: Thêm early stopping và validation parameters cho MLP

## Cấu trúc dự án ✨ **REORGANIZED**

### 📁 **Cấu trúc thư mục mới (v2.0)**

```
ml/
├── 📁 config/                     # Cấu hình YAML
│   ├── data_config.yaml           # Cấu hình dữ liệu
│   ├── model_config.yaml          # Cấu hình model parameters
│   ├── optimization_config.yaml   # Cấu hình algorithms
│   └── model_params.py            # Legacy parameters (backward compatibility)
│
├── 📁 data/                       # Data processing
│   ├── __init__.py
│   └── data_preprocessing.py      # Data loading & preprocessing
│
├── 📁 models/                     # Model definitions
│   └── __init__.py                # (Chuẩn bị cho model classes)
│
├── 📁 optimization/               # Optimization algorithms
│   ├── __init__.py
│   ├── pso_optimizer.py           # PSO algorithm
│   ├── rso_optimizer.py           # RSO algorithm  
│   └── puma_optimizer.py          # PUMA algorithm
│
├── 📁 evaluation/                 # Evaluation & metrics
│   ├── __init__.py
│   └── evaluation_utils.py        # Metrics calculation
│
├── 📁 utils/                      # Utilities
│   ├── __init__.py
│   ├── config.py                  # Legacy config (backward compatibility)
│   ├── yaml_config.py             # YAML config management ✨ *New*
│   └── logging.py                 # Logging utilities ✨ *New*
│
├── 📁 experiments/               # Experiment runners
│   ├── __init__.py
│   ├── pso_rf.py                 # PSO + Random Forest
│   ├── pso_svm.py                # PSO + SVM
│   ├── pso_xgb.py                # PSO + XGBoost
│   ├── pso_mlp.py                # PSO + MLP
│   ├── rso_rf.py                 # RSO + Random Forest
│   ├── rso_svm.py                # RSO + SVM
│   ├── rso_xgb.py                # RSO + XGBoost
│   ├── rso_mlp.py                # RSO + MLP
│   ├── po_rf.py                  # PUMA + Random Forest
│   ├── po_svm.py                 # PUMA + SVM
│   ├── po_xgb.py                 # PUMA + XGBoost
│   └── po_mlp.py                 # PUMA + MLP
│
├── 📁 scripts/                   # Entry point scripts
│   └── run_optimization.py       # Main runner script ✨ *New*
│
├── __init__.py                   # Package initialization ✨ *New*
├── requirements.txt              # Dependencies ✨ *New*
└── README.md                     # Documentation
```

### 🆕 **Cải tiến chính (v2.0)**

#### ✅ **Modular Architecture**
- **Separation of Concerns**: Tách biệt data, models, optimization, evaluation
- **Clear Dependencies**: Mỗi module có trách nhiệm rõ ràng
- **Easy Testing**: Structure cho phép unit testing dễ dàng

#### ✅ **YAML Configuration Management**
- **Centralized Config**: Tất cả config tập trung trong `config/`
- **Environment Flexible**: Dễ dàng switch giữa các environment
- **Backward Compatible**: Vẫn support config cũ

#### ✅ **Enhanced Developer Experience**
- **Entry Point Scripts**: `scripts/run_optimization.py` làm entry point chính
- **Proper Logging**: Structured logging với levels
- **Package Structure**: Proper Python package với `__init__.py`

## Chi tiết thuật toán

### 🔵 Particle Swarm Optimization (PSO)
**Nguyên lý**: Mô phỏng hành vi bầy đàn của chim/cá để tìm kiếm tối ưu.

**Tham số chính**:
- `n_particles`: Số lượng particles (mặc định: 10)
- `n_iterations`: Số lần lặp (mặc định: 100)
- `w`: Inertia weight (0.5)
- `c1, c2`: Acceleration coefficients (2.0, 2.0)

**Ưu điểm**:
- Hội tụ nhanh
- Khả năng thoát khỏi local optima tốt
- Phù hợp với không gian tìm kiếm liên tục

### 🟢 Random Search Optimization (RSO)
**Nguyên lý**: Tạo ngẫu nhiên các tham số và chọn ra tốt nhất.

**Tham số chính**:
- `n_iterations`: Số lần lặp tìm kiếm (mặc định: 100)

**Ưu điểm**:
- Đơn giản, dễ hiểu và triển khai
- Không bị mắc kẹt trong local optima
- Hiệu quả với không gian tham số lớn

### 🟡 PUMA Optimization ✨ *Updated*
**Nguyên lý**: Thuật toán meta-heuristic tiên tiến với 2 giai đoạn: Exploration và Exploitation.

**Tham số chính**:
- `population_size`: Kích thước quần thể (mặc định: 10)
- `generations`: Số thế hệ (mặc định: 100)
- `pCR`: Crossover rate tự động điều chỉnh (0.5)

**Đặc điểm**:
- Adaptive phase switching dựa trên scoring system
- Tự động cân bằng giữa khám phá và khai thác
- Hỗ trợ đầy đủ cho MLP với metrics chính xác

**Cải thiện mới**: 
- Loại bỏ hiển thị Phase column để tập trung vào metrics quan trọng
- Custom metrics calculation cho từng loại model

## Chi tiết mô hình

### 🌲 Random Forest
**Tham số tối ưu hóa**:
- `n_estimators`: 50-500 cây
- `max_depth`: 3-20 hoặc None
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10
- `max_features`: ['auto', 'sqrt', 'log2']
- `bootstrap`: [True, False]

### 🎯 Support Vector Machine (SVM)
**Tham số tối ưu hóa**:
- `C`: 0.1-1000 (log-uniform)
- `gamma`: ['scale', 'auto'] hoặc 0.001-1
- `kernel`: ['linear', 'poly', 'rbf', 'sigmoid']
- `degree`: 2-5 (cho poly kernel)
- `epsilon`: 0.01-1.0

### 🚀 XGBoost
**Tham số tối ưu hóa**:
- `n_estimators`: 50-500
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.3
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `gamma`: 0-5
- `reg_alpha`: 0-1
- `reg_lambda`: 0-1

### 🧠 Multi-Layer Perceptron (MLP) ✨ *Enhanced*
**Tham số tối ưu hóa** (13 parameters):

**Architecture**:
- `hidden_layer_sizes`: 12 kiến trúc được định nghĩa sẵn từ (50,) đến (200,100,50)

**Training**:
- `activation`: ['relu', 'tanh', 'logistic'] - Hàm kích hoạt
- `solver`: ['adam', 'lbfgs', 'sgd'] - Thuật toán tối ưu hóa
- `alpha`: 0.0001-0.01 (log-uniform) - L2 regularization
- `learning_rate`: ['constant', 'invscaling', 'adaptive'] - Learning rate schedule
- `learning_rate_init`: 0.001-0.1 (log-uniform) - Initial learning rate

**Adam Optimizer**:
- `beta_1`: 0.8-0.999 - Exponential decay rate for 1st moment
- `beta_2`: 0.9-0.9999 - Exponential decay rate for 2nd moment  
- `epsilon`: 1e-9 to 1e-7 (log-uniform) - Numerical stability

**Early Stopping** ✨ *New*:
- `validation_fraction`: 0.05-0.3 - Tỷ lệ dữ liệu validation
- `n_iter_no_change`: 5-20 - Số epochs không cải thiện trước khi dừng
- `tol`: 1e-6 to 1e-3 (log-uniform) - Tolerance for optimization
- `max_iter`: 100-1000 - Maximum iterations

**Cải thiện**: Thêm đầy đủ early stopping parameters giúp model ổn định và tránh overfitting.

## Hàm đánh giá (Fitness Function)

### 📊 Regression Metrics
```python
fitness = R² - RMSE - MAE
```

**Giải thích**:
- **R² (R-squared)**: Hệ số xác định (0-1, càng cao càng tốt)
- **RMSE (Root Mean Square Error)**: Sai số bình phương trung bình (càng thấp càng tốt)
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình (càng thấp càng tốt)

**Mục tiêu**: Maximization (tối đa hóa fitness)

## Cách sử dụng ✨ **ENHANCED**

### 1. Cài đặt dependencies
```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt manual
pip install numpy pandas scikit-learn xgboost PyYAML
```

### 2. **Cách sử dụng mới (v2.0)** ✨

#### Sử dụng script runner chính:
```bash
# PSO cho Random Forest  
python scripts/run_optimization.py --model rf --optimizer pso

# RSO cho SVM
python scripts/run_optimization.py --model svm --optimizer rso

# PUMA cho XGBoost
python scripts/run_optimization.py --model xgb --optimizer puma

# MLP với verbose output
python scripts/run_optimization.py --model mlp --optimizer pso --verbose
```

#### Import như Python package:
```python
# Import framework
from ml.optimization import PSOOptimizer, PUMAOptimizer
from ml.utils.yaml_config import load_model_config, get_optimization_params
from ml.data import load_flood_data

# Load data
X_train, X_test, y_train, y_test = load_flood_data()

# Load config
pso_config = get_optimization_params('pso')
model_config = get_model_config('rf')

# Run optimization
optimizer = PSOOptimizer(X_train, y_train, **pso_config)
best_params, best_score = optimizer.optimize()
```

### 3. **Cách sử dụng cũ (Backward Compatible)**

#### Chạy trực tiếp từ experiments/:
```bash
cd experiments/

# PSO cho Random Forest
python pso_rf.py

# RSO cho SVM  
python rso_svm.py

# PUMA cho MLP
python po_mlp.py
```

## Cấu hình dữ liệu

### 📊 Định dạng dữ liệu
Dữ liệu đầu vào phải ở dạng CSV với các cột sau:

**Features (Đặc trưng)**:
- `Rainfall`: Lượng mưa
- `Elevation`: Độ cao
- `Slope`: Độ dốc
- `Aspect`: Hướng sườn
- `Flow_direction`: Hướng dòng chảy
- `Flow_accumulation`: Tích lũy dòng chảy
- `TWI`: Topographic Wetness Index
- `Distance_to_river`: Khoảng cách đến sông
- `Drainage_capacity`: Khả năng thoát nước
- `LandCover`: Lớp phủ đất
- `Imperviousness`: Độ không thấm
- `Surface_temperature`: Nhiệt độ bề mặt

**Target (Mục tiêu)**:
- Giá trị liên tục từ 0-1 (xác suất lũ lụt)

### 🔧 Cấu hình trong `config.py`
```python
DATA_CONFIG = {
    'file_path': 'đường/dẫn/đến/file.csv',
    'feature_columns': [...],
    'target_column': 'target_name'
}

OPTIMIZATION_CONFIG = {
    'n_particles': 10,
    'n_iterations': 100,
    'population_size': 30,
    'generations': 100
}
```

## Kết quả đầu ra

### 📈 Trong quá trình chạy
```
Lần lặp 1/100
Điểm số hiện tại: 0.850000
Điểm số tốt nhất: 0.850000
R²: 0.823456 | RMSE: 0.156789 | MAE: 0.123456
```

### 📋 Kết quả cuối cùng
```
=== Kết quả cuối cùng ===
Điểm số tốt nhất: 0.892345
Tham số tối ưu:
  n_estimators: 200
  max_depth: 8
  min_samples_split: 5
  ...
```

### 📁 Files đầu ra
- `{algorithm}_{model}_metrics.csv`: Metrics cuối cùng
- Console logs với detailed progress

## Tùy chỉnh nâng cao

### 🔧 Thêm mô hình mới
1. Tạo class optimizer mới kế thừa từ base optimizer
2. Implement `_create_model()` và `evaluate_model()`
3. Định nghĩa param_ranges trong `model_params.py`

### 📊 Thêm metrics mới
Chỉnh sửa `evaluate_regression_model()` trong `evaluation_utils.py`:
```python
def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    # Thêm metrics mới
    new_metric = calculate_new_metric(y_test, y_pred)
    return r2 - rmse - mae + new_metric  # Adjust formula
```

### ⚙️ Thay đổi thuật toán tối ưu hóa
Tạo optimizer mới trong file riêng:
```python
class MyOptimizer:
    def __init__(self, X, y, **kwargs):
        # Implementation
    
    def optimize(self):
        # Optimization logic
```

## Troubleshooting

### ❌ Lỗi thường gặp

#### 1. FileNotFoundError
```
Lỗi: Không tìm thấy file dữ liệu!
```
**Giải pháp**: Kiểm tra đường dẫn file trong `config.py` và `data_preprocessing.py`

#### 2. MLP Metrics không đồng bộ ✅ *Fixed*
```
Gen | Fitness     | R²        | MAE       | RMSE
  1 |   0.827445 | 0.792969 |   0.1811 |   0.2245
  2 |   0.843924 | 0.792969 |   0.1811 |   0.2245  # Same metrics!
```
**Đã sửa**: Tạo `calculate_mlp_metrics()` riêng trong `po_mlp.py` để đảm bảo metrics tính từ đúng model.

#### 3. Config Duplication Issues ✅ *Fixed*
```
ImportError: Duplicate OPTIMIZATION_CONFIG found
```
**Đã sửa**: Tách biệt config trong `config.py` và `model_params.py`, loại bỏ duplicate.

#### 4. MLP Early Stopping không hoạt động ✅ *Fixed*
```
MLP không cải thiện sau 10+ generations
```
**Đã sửa**: Thêm đầy đủ early stopping parameters (`validation_fraction`, `n_iter_no_change`, `tol`).

#### 2. Memory Error
```
MemoryError: Unable to allocate array
```
**Giải pháp**: 
- Giảm kích thước dữ liệu
- Giảm số particles/iterations
- Sử dụng `n_jobs=1` thay vì `-1`

#### 3. Convergence Warning
```
ConvergenceWarning: lbfgs failed to converge
```
**Giải pháp**:
- Tăng `max_iter` cho MLP
- Thay đổi solver (adam thay vì lbfgs)
- Chuẩn hóa dữ liệu tốt hơn

#### 4. Poor Performance
**Giải pháp**:
- Tăng số iterations/generations
- Kiểm tra chất lượng dữ liệu
- Thử thuật toán tối ưu khác
- Điều chỉnh param_ranges

### 🔍 Debug Tips
1. **Kiểm tra dữ liệu**: Verify data shape, missing values, range
2. **Test với dữ liệu nhỏ**: Sử dụng subset để debug nhanh
3. **Monitor metrics**: Quan sát xu hướng R², RMSE, MAE
4. **Compare algorithms**: So sánh PSO vs RSO vs PUMA

## Performance Guidelines

### 🚀 Optimization Tips
1. **Số lượng particles/iterations**:
   - Bắt đầu với 10 particles, 50 iterations
   - Tăng dần nếu kết quả chưa hội tụ
   - PSO: 10-30 particles, 50-200 iterations
   - RSO: 100-500 iterations
   - PUMA: 20-50 population, 50-200 generations

2. **Parallel processing**:
   - Sử dụng `n_jobs=-1` cho Random Forest, XGBoost
   - Tránh parallel trong cross-validation với SVM
   - Monitor CPU/memory usage

3. **Data preprocessing**:
   - Chuẩn hóa features (StandardScaler)
   - Handle missing values properly
   - Remove outliers if necessary

### ⏱️ Estimated Runtime
| Model | Algorithm | Data Size | Time (approx) |
|-------|-----------|-----------|---------------|
| RF | PSO | 1000 samples | 2-5 minutes |
| SVM | RSO | 1000 samples | 5-15 minutes |
| XGBoost | PUMA | 1000 samples | 3-8 minutes |
| MLP | PSO | 1000 samples | 5-20 minutes |

## Khuyến nghị sử dụng

### 🎯 Lựa chọn thuật toán
- **PSO**: Tốt cho hầu hết các trường hợp, hội tụ nhanh
- **RSO**: Dùng khi cần đơn giản, không gian tham số lớn
- **PUMA**: Cho các bài toán phức tạp, cần khám phá sâu

### 🤖 Lựa chọn mô hình
- **Random Forest**: Đa năng, robust, dễ interpret
- **SVM**: Tốt với dữ liệu chiều cao, non-linear patterns
- **XGBoost**: High performance, competition-grade
- **MLP**: Deep patterns, complex relationships

### 📊 Đánh giá kết quả
- **R² > 0.8**: Excellent
- **R² 0.6-0.8**: Good
- **R² 0.4-0.6**: Moderate
- **R² < 0.4**: Poor (cần review data/model)

## Contributing

### 🤝 Hướng dẫn đóng góp
1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### 📝 Code Style
- Sử dụng type hints
- Docstrings cho functions/classes
- Follow PEP 8
- Comprehensive error handling

## License & Contact

**License**: [Specify your license]

**Authors**: [Your team info]

**Contact**: [Contact information]

---

## Changelog

### 🆕 Version 2.1.0 (Latest)
- ✅ **MLP Optimization Enhancement**: Thêm đầy đủ 13 tham số cho MLPRegressor
- ✅ **Fixed PUMA-MLP Metrics Issue**: Tạo `calculate_mlp_metrics()` để đảm bảo metrics đồng bộ với fitness
- ✅ **UI Improvement**: Loại bỏ cột "Phase" khỏi output table, tập trung vào metrics quan trọng
- ✅ **Config Reorganization**: Tách biệt config general và model-specific, loại bỏ duplication
- ✅ **Stability Improvements**: Thêm early stopping parameters cho MLP

### 📋 Version 2.0.0
- 🔄 **PUMA Algorithm Integration**: Thêm PUMA optimizer cho tất cả models
- 🔄 **Unified Framework**: Chuẩn hóa interface cho tất cả optimizers
- 🔄 **Enhanced Evaluation**: Cải thiện fitness function và metrics calculation

### 📋 Version 1.0.0  
- 🎉 **Initial Release**: PSO và RSO cho Random Forest, SVM, XGBoost, MLP
- 🎉 **Basic Framework**: Core optimization infrastructure

### Version 1.0 (Current)
- ✅ PSO, RSO, PUMA optimizers
- ✅ RF, SVM, XGBoost, MLP models
- ✅ Unified fitness function
- ✅ Comprehensive utilities
- ✅ Detailed logging and metrics

### Planned Features
- 🔲 GUI interface
- 🔲 More optimization algorithms
- 🔲 Ensemble methods
- 🔲 Automated hyperparameter tuning
- 🔲 Model comparison dashboard
- 🔲 Export to production formats

---

> **Note**: Đây là framework nghiên cứu. Khi triển khai production, cần bổ sung validation, security, và monitoring.
