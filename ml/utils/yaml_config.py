"""Configuration management utilities với YAML support"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Backward compatibility - giữ lại cấu hình cũ
from .config import *

# Đường dẫn tới thư mục config
CONFIG_DIR = Path(__file__).parent.parent / "config"

def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration từ file YAML
    
    Args:
        config_name: Tên file config (không cần .yaml extension)
        
    Returns:
        Dictionary chứa configuration
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data_config() -> Dict[str, Any]:
    """Load data configuration từ YAML"""
    try:
        return load_yaml_config("data_config")
    except FileNotFoundError:
        # Fallback to old config
        return {
            'data': {
                'flood_csv_path': DATA_CONFIG['flood_csv_path'],
                'preprocessing': {
                    'separator': FLOOD_DATA_CONFIG['separator'],
                    'na_values': FLOOD_DATA_CONFIG['na_values'],
                    'test_size': FLOOD_DATA_CONFIG['test_size'],
                    'random_state': FLOOD_DATA_CONFIG['random_state'],
                    'imputation_strategy': FLOOD_DATA_CONFIG['imputation_strategy']
                },
                'features': FLOOD_DATA_CONFIG['feature_columns'],
                'target': {
                    'column': FLOOD_DATA_CONFIG['label_column'],
                    'mapping': FLOOD_DATA_CONFIG['target_mapping']
                }
            },
            'output': {
                'base_dir': DATA_CONFIG['output_dir'],
                'models_dir': DATA_CONFIG['models_dir'],
                'results_dir': DATA_CONFIG['results_dir']
            }
        }

def load_model_config() -> Dict[str, Any]:
    """Load model configuration từ YAML"""
    try:
        return load_yaml_config("model_config")
    except FileNotFoundError:
        # Fallback to old config
        from ..config.model_params import MODEL_PARAM_RANGES
        return MODEL_PARAM_RANGES

def load_optimization_config() -> Dict[str, Any]:
    """Load optimization configuration từ YAML"""
    try:
        return load_yaml_config("optimization_config")
    except FileNotFoundError:
        # Fallback to old config  
        from ..config.model_params import OPTIMIZATION_CONFIG
        return {
            'pso': {
                'population_size': OPTIMIZATION_CONFIG.get('n_particles', 10),
                'n_iterations': OPTIMIZATION_CONFIG.get('n_iterations', 100),
                'inertia_weight': 0.5,
                'cognitive_coeff': 2.0,
                'social_coeff': 2.0
            },
            'rso': {
                'n_iterations': 100
            },
            'puma': {
                'population_size': OPTIMIZATION_CONFIG.get('population_size', 10),
                'generations': OPTIMIZATION_CONFIG.get('generations', 100)
            }
        }

def get_yaml_output_path(filename: str, subdir: str = "base_dir") -> str:
    """
    Lấy đường dẫn file output từ YAML config
    """
    try:
        data_config = load_data_config()
        output_config = data_config.get("output", {})
        base_path = output_config.get(subdir, "./outputs")
    except:
        # Fallback to old method
        return get_output_path(filename, subdir)
    
    # Tạo thư mục nếu chưa tồn tại
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    return os.path.join(base_path, filename)

def get_model_param_ranges(model_name: str) -> Dict[str, Any]:
    """
    Lấy parameter ranges cho model cụ thể từ YAML config
    """
    try:
        model_config = load_model_config()
        
        # Mapping tên model
        model_mapping = {
            'rf': 'random_forest',
            'random_forest': 'random_forest',
            'xgb': 'xgboost', 
            'xgboost': 'xgboost',
            'svm': 'svm',
            'support_vector_machine': 'svm',
            'mlp': 'mlp',
            'neural_network': 'mlp'
        }
        
        actual_model_name = model_mapping.get(model_name.lower())
        if not actual_model_name:
            supported_models = list(model_mapping.keys())
            raise ValueError(f"Model '{model_name}' không được hỗ trợ. "
                            f"Các model được hỗ trợ: {supported_models}")
        
        model_info = model_config.get(actual_model_name)
        if not model_info:
            raise ValueError(f"Không tìm thấy config cho model: {actual_model_name}")
        
        return model_info.get("param_ranges", {})
        
    except:
        # Fallback to old method
        from ..config.model_params import get_param_ranges
        return get_param_ranges(model_name)

def get_optimization_params(algorithm: str) -> Dict[str, Any]:
    """
    Lấy parameters cho optimization algorithm từ YAML config
    """
    try:
        opt_config = load_optimization_config()
        
        if algorithm.lower() not in opt_config:
            supported_algs = list(opt_config.keys())
            raise ValueError(f"Algorithm '{algorithm}' không được hỗ trợ. "
                            f"Các algorithm được hỗ trợ: {supported_algs}")
        
        return opt_config[algorithm.lower()]
        
    except:
        # Fallback to default values
        default_configs = {
            'pso': {
                'population_size': 10,
                'n_iterations': 100,
                'inertia_weight': 0.5,
                'cognitive_coeff': 2.0,
                'social_coeff': 2.0
            },
            'rso': {
                'n_iterations': 100
            },
            'puma': {
                'population_size': 10,
                'generations': 100
            }
        }
        return default_configs.get(algorithm.lower(), {})