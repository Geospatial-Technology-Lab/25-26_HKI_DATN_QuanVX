"""
Hyperparameter configurations for ML models
Cấu hình tham số tối ưu cho các model ML
Format chuẩn để sử dụng trong code, nhưng hiện tại không dùng
"""

# Random Forest Parameters
RF_PARAMS = {
    'pso_rf': {
        'n_estimators': 1000,
        'max_depth': 50,
        'min_samples_split': 20,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': False,
        'max_leaf_nodes': 1000,
    },
    
    'po_rf': {
        'n_estimators': 50,
        'max_depth': 14,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'log2',
        'bootstrap': False,
        'max_leaf_nodes': 879,
    },
    
    'rso_rf': {
        'n_estimators': 459,
        'max_depth': 16,
        'min_samples_split': 13,
        'min_samples_leaf': 9,
        'max_features': 'sqrt',
        'bootstrap': False,
        'max_leaf_nodes': 911,
    }
}

# SVM Parameters
SVM_PARAMS = {
    'pso_svm': {
        'C': 1.19288,
        'gamma': 0.078738,
        'kernel': 'poly',
        'degree': 4,
        'coef0': 10,
        'tol': 1e-05,
        'epsilon': 0.1178,
        'max_iter': 42540,
        'shrinking': True
    },
    
    'po_svm': {
        'C': 566.6982,
        'gamma': 0.153971,
        'kernel': 'rbf',
        'degree': 5,
        'coef0': 0,
        'tol': 0.001717,
        'epsilon': 0.01,
        'max_iter': 50000,
        'shrinking': True
    },
    
    'rso_svm': {
        'C': 0.001235,
        'gamma': 4.647095,
        'kernel': 'poly',
        'degree': 5,
        'coef0': 1.76137,
        'tol': 0.000319,
        'epsilon': 0.526529,
        'max_iter': 43628,
        'shrinking': False
    }
}

# XGBoost Parameters
XGB_PARAMS = {
    'pso_xgb': {
        'n_estimators': 1000,
        'max_depth': 15,
        'learning_rate': 0.01,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.9549,
        'colsample_bynode': 1.0,
        'reg_alpha': 0.059774,
        'reg_lambda': 1.0,
        'min_child_weight': 1,
        'gamma': 0,
        'max_delta_step': 10,
        'scale_pos_weight': 0.5
    },
    
    'po_xgb': {
        'n_estimators': 813,
        'max_depth': 15,
        'learning_rate': 0.01,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 0.995257,
        'reg_alpha': 0.166811,
        'reg_lambda': 0.718503,
        'min_child_weight': 1,
        'gamma': 0,
        'max_delta_step': 6,
        'scale_pos_weight': 0.5
    },
    
    'rso_xgb': {
        'n_estimators': 460,
        'max_depth': 7,
        'learning_rate': 0.029195,
        'subsample': 0.096524,
        'colsample_bytree': 0.826872,
        'colsample_bylevel': 0.88716,
        'colsample_bynode': 0.68051,
        'reg_alpha': 0.499231,
        'reg_lambda': 0.884683,
        'min_child_weight': 8,
        'gamma': 0.714358,
        'max_delta_step': 2,
        'scale_pos_weight': 0.8699
    }
}

# MLP Parameters (if needed)
MLP_PARAMS = {
    'pso_mlp': {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_iter': 1000
    },
    
    'po_mlp': {
        'hidden_layer_sizes': (150, 75),
        'activation': 'tanh',
        'solver': 'lbfgs',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.01,
        'max_iter': 500
    },
    
    'rso_mlp': {
        'hidden_layer_sizes': (200, 100, 50),
        'activation': 'logistic',
        'solver': 'sgd',
        'alpha': 0.01,
        'learning_rate': 'invscaling',
        'learning_rate_init': 0.1,
        'max_iter': 2000
    }
}

# Utility function to get parameters
def get_model_params(model_type: str, optimization_method: str) -> dict:
    """
    Get parameters for specific model and optimization method
    
    Args:
        model_type: 'rf', 'svm', 'xgb', 'mlp'
        optimization_method: 'pso', 'po', 'rso'
    
    Returns:
        dict: Model parameters
    """
    param_map = {
        'rf': RF_PARAMS,
        'svm': SVM_PARAMS,
        'xgb': XGB_PARAMS,
        'mlp': MLP_PARAMS
    }
    
    key = f"{optimization_method}_{model_type}"
    
    if model_type in param_map and key in param_map[model_type]:
        return param_map[model_type][key].copy()
    else:
        print(f"⚠️ No parameters found for {key}")
        return {}

# Example usage (commented out - not in use):
"""
# To use these parameters:
from ml_hyper_parameter import get_model_params

# Get PSO-optimized Random Forest parameters
rf_params = get_model_params('rf', 'pso')
model = RandomForestClassifier(**rf_params)

# Get PO-optimized SVM parameters  
svm_params = get_model_params('svm', 'po')
model = SVC(**svm_params)

# Get RSO-optimized XGBoost parameters
xgb_params = get_model_params('xgb', 'rso') 
model = XGBClassifier(**xgb_params)
"""