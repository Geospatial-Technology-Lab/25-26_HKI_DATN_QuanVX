import joblib
from pathlib import Path

def load_all_models(model_dir):
    """Load tất cả models từ thư mục"""
    models = {}
    model_dir = Path(model_dir)
    
    for model_file in list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl")):
        models[model_file.stem] = joblib.load(model_file)
    
    return models