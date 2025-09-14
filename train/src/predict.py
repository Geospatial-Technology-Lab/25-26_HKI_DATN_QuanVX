"""Simple flood prediction using data_utils workflow."""

from pathlib import Path
from data_utils import load_models, predict_to_tiff


def main():
    """Run flood prediction using clean workflow."""
    print("🚀 Flood Prediction with Config-based Workflow")
    
    # Load models
    model_dir = Path(__file__).parent.parent / "model"
    models = load_models(model_dir)
    
    if not models:
        print("❌ No models found!")
        return
    
    # Run prediction to TIFF
    data_file = Path(r"D:\QuanVX\QuanVX\Default.gdb\a000000c8.gdbtable")
    output_dir = Path(__file__).parent.parent / "results"
    layer_name = "RasterT_Extract1"
    
    predict_to_tiff(
        models=models,
        file_path=data_file, 
        layer_name=layer_name,
        output_dir=output_dir,
        chunk_size=50000
    )
    
    print("🎉 Prediction completed!")


if __name__ == "__main__":
    main()