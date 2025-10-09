"""Các hàm tiện ích chung cho tạo và xử lý mô hình ML"""

import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


def generate_random_params(param_ranges):
    """Tạo tham số ngẫu nhiên dựa trên param_ranges"""
    params = {}
    for param, range_info in param_ranges.items():
        if range_info['type'] == 'int':
            params[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
        elif range_info['type'] == 'float':
            params[param] = np.random.uniform(range_info['min'], range_info['max'])
        elif range_info['type'] == 'log_uniform':
            log_min = np.log10(range_info['min'])
            log_max = np.log10(range_info['max'])
            params[param] = 10 ** np.random.uniform(log_min, log_max)
        elif range_info['type'] == 'choice':
            params[param] = random.choice(range_info['options'])
    return params


def clean_svm_params(params):
    """Loại bỏ tham số không tương thích với kernel của SVM"""
    kernel = params.get('kernel', 'rbf')
    if kernel == 'linear':
        params.pop('gamma', None)
        params.pop('degree', None)
        params.pop('coef0', None)
    elif kernel == 'rbf':
        params.pop('degree', None)
        params.pop('coef0', None)
    elif kernel == 'sigmoid':
        params.pop('degree', None)
    return params


def clean_mlp_params(params):
    """Loại bỏ tham số không tương thích với solver của MLP"""
    solver = params.get('solver', 'adam')
    if solver == 'lbfgs':
        for param in ['learning_rate_init', 'learning_rate', 'beta_1', 'beta_2', 'epsilon']:
            params.pop(param, None)
    elif solver == 'sgd':
        for param in ['beta_1', 'beta_2', 'epsilon']:
            params.pop(param, None)
    return params


def create_model(model_name, params, random_state=42):
    """Tạo model dựa trên tên và tham số"""
    model_name = model_name.lower()
    
    if model_name in ['rf', 'random_forest']:
        return RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
    
    elif model_name in ['xgb', 'xgboost']:
        xgb_params = params.copy()
        xgb_params.update({'random_state': random_state, 'n_jobs': -1, 'verbosity': 0})
        return xgb.XGBRegressor(**xgb_params)
    
    elif model_name in ['svm', 'support_vector_machine']:
        svm_params = clean_svm_params(params.copy())
        return SVR(**svm_params)
    
    elif model_name in ['mlp', 'neural_network', 'multi_layer_perceptron']:
        mlp_params = clean_mlp_params(params.copy())
        mlp_params.update({'random_state': random_state, 'early_stopping': True})
        return MLPRegressor(**mlp_params)
    
    else:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ")
