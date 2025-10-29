RF_PARAM_RANGES = {
    'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
    'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
    'max_features': {'type': 'choice', 'options': ['sqrt', 'log2', None]},
    'bootstrap': {'type': 'choice', 'options': [True, False]}
}

XGB_PARAM_RANGES = {
    'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
    'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
    'subsample': {'type': 'float', 'min': 0.6, 'max': 1.0},
    'colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0},
    'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 1.0}
}

SVM_PARAM_RANGES = {
    'C': {'type': 'log_uniform', 'min': 0.001, 'max': 1000},
    'gamma': {'type': 'log_uniform', 'min': 0.0001, 'max': 10},
    'kernel': {'type': 'choice', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
    'degree': {'type': 'int', 'min': 2, 'max': 5},
    'coef0': {'type': 'float', 'min': 0.0, 'max': 10.0}
}

MODEL_PARAM_RANGES = {
    'rf': RF_PARAM_RANGES,
    'random_forest': RF_PARAM_RANGES,
    'xgb': XGB_PARAM_RANGES,
    'xgboost': XGB_PARAM_RANGES,
    'svm': SVM_PARAM_RANGES,
    'support_vector_machine': SVM_PARAM_RANGES
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
    'RF_PARAM_RANGES', 'XGB_PARAM_RANGES', 'SVM_PARAM_RANGES',
    'MODEL_PARAM_RANGES', 'get_param_ranges', 'get_all_supported_models'
]