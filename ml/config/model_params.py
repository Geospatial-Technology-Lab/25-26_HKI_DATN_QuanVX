RF_PARAM_RANGES = {
    'numberOfTrees': {'type': 'int', 'min': 50, 'max': 1000},
    'variablesPerSplit': {'type': 'int', 'min': 1, 'max': 13},
    'minLeafPopulation': {'type': 'int', 'min': 1, 'max': 20},
    'bagFraction': {'type': 'float', 'min': 0.4, 'max': 1.0},
    'maxNodes': {'type': 'int', 'min': 50, 'max': 20000}
}

XGB_PARAM_RANGES = {
    'numberOfTrees': {'type': 'int', 'min': 100, 'max': 1500},
    'shrinkage': {'type': 'float', 'min': 0.001, 'max': 0.5},
    'samplingRate': {'type': 'float', 'min': 0.5, 'max': 1.0},
    'maxNodes': {'type': 'int', 'min': 50, 'max': 20000},
    'loss': {'type': 'choice', 'options': ['LeastAbsoluteDeviation', 'LeastSquares', 'Huber']}
}

SVM_PARAM_RANGES = {
    'cost': {'type': 'log_uniform', 'min': 0.01, 'max': 200},
    'gamma': {'type': 'log_uniform', 'min': 0.0001, 'max': 2.0},
    'epsilon': {'type': 'float', 'min': 0.001, 'max': 0.5},
    'kernelType': {'type': 'choice', 'options': ['RBF', 'POLY', 'LINEAR']},
    'degree': {'type': 'int', 'min': 2, 'max': 5},
    'coef0': {'type': 'float', 'min': 0.0, 'max': 20.0}
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