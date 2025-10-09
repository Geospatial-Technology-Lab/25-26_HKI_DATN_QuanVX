"""File cấu hình tham số cho các mô hình Machine Learning"""

# Tham số cho Random Forest
RF_PARAM_RANGES = {
    'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
    'max_depth': {'type': 'int', 'min': 3, 'max': 50},
    'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
    'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
    'max_features': {'type': 'choice', 'options': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9]},
    'bootstrap': {'type': 'choice', 'options': [True, False]},
    'max_leaf_nodes': {'type': 'int', 'min': 10, 'max': 1000},
    'min_impurity_decrease': {'type': 'float', 'min': 0.0, 'max': 0.2}
}

# Tham số cho XGBoost
XGB_PARAM_RANGES = {
    'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
    'max_depth': {'type': 'int', 'min': 3, 'max': 15},
    'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
    'subsample': {'type': 'float', 'min': 0.6, 'max': 1.0},
    'colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0},
    'colsample_bylevel': {'type': 'float', 'min': 0.6, 'max': 1.0},
    'colsample_bynode': {'type': 'float', 'min': 0.6, 'max': 1.0},
    'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'min_child_weight': {'type': 'int', 'min': 1, 'max': 10},
    'gamma': {'type': 'float', 'min': 0.0, 'max': 5.0},
    'max_delta_step': {'type': 'int', 'min': 0, 'max': 10},
    'scale_pos_weight': {'type': 'float', 'min': 0.5, 'max': 2.0}
}

# Tham số cho SVR (Support Vector Regression)
SVM_PARAM_RANGES = {
    'C': {'type': 'log_uniform', 'min': 0.001, 'max': 1000},
    'gamma': {'type': 'log_uniform', 'min': 0.0001, 'max': 10},
    'kernel': {'type': 'choice', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
    'degree': {'type': 'int', 'min': 2, 'max': 5},
    'coef0': {'type': 'float', 'min': 0.0, 'max': 10.0},
    'tol': {'type': 'log_uniform', 'min': 1e-5, 'max': 1e-2},
    'epsilon': {'type': 'float', 'min': 0.01, 'max': 1.0},
    'max_iter': {'type': 'int', 'min': 1000, 'max': 50000},
    'shrinking': {'type': 'choice', 'options': [True, False]}
}

# Tham số cho MLP (Multi-Layer Perceptron)
MLP_PARAM_RANGES = {
    'hidden_layer_sizes': {'type': 'choice', 'options': [
        (50,), (100,), (150,), (200,), 
        (50, 50), (100, 100), (150, 150),
        (100, 50), (150, 100), (200, 100),
        (100, 50, 25), (150, 100, 50), (200, 150, 100)
    ]},
    'activation': {'type': 'choice', 'options': ['relu', 'tanh', 'logistic', 'identity']},
    'solver': {'type': 'choice', 'options': ['adam', 'sgd', 'lbfgs']},
    'alpha': {'type': 'log_uniform', 'min': 1e-5, 'max': 1e-1},
    'learning_rate': {'type': 'choice', 'options': ['constant', 'invscaling', 'adaptive']},
    'learning_rate_init': {'type': 'log_uniform', 'min': 1e-4, 'max': 1e-1},
    'max_iter': {'type': 'int', 'min': 200, 'max': 1000},
    'beta_1': {'type': 'float', 'min': 0.8, 'max': 0.99},
    'beta_2': {'type': 'float', 'min': 0.9, 'max': 0.999},
    'epsilon': {'type': 'log_uniform', 'min': 1e-9, 'max': 1e-6},
    'validation_fraction': {'type': 'float', 'min': 0.1, 'max': 0.3},
    'n_iter_no_change': {'type': 'int', 'min': 10, 'max': 50},
    'tol': {'type': 'log_uniform', 'min': 1e-5, 'max': 1e-2}
}

# Dictionary để dễ dàng truy cập tham số theo tên mô hình
MODEL_PARAM_RANGES = {
    'rf': RF_PARAM_RANGES,
    'random_forest': RF_PARAM_RANGES,
    'xgb': XGB_PARAM_RANGES,
    'xgboost': XGB_PARAM_RANGES,
    'svm': SVM_PARAM_RANGES,
    'support_vector_machine': SVM_PARAM_RANGES,
    'mlp': MLP_PARAM_RANGES,
    'multi_layer_perceptron': MLP_PARAM_RANGES,
    'neural_network': MLP_PARAM_RANGES
}

def get_param_ranges(model_name):
    model_name = model_name.lower()
    if model_name not in MODEL_PARAM_RANGES:
        supported_models = list(set(MODEL_PARAM_RANGES.keys()))
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. "
                        f"Các mô hình được hỗ trợ: {supported_models}")
    
    return MODEL_PARAM_RANGES[model_name]

def get_all_supported_models():
    return list(set(MODEL_PARAM_RANGES.keys()))
__all__ = [
    'RF_PARAM_RANGES', 'XGB_PARAM_RANGES', 'SVM_PARAM_RANGES', 'MLP_PARAM_RANGES',
    'MODEL_PARAM_RANGES', 'get_param_ranges', 'get_all_supported_models'
]