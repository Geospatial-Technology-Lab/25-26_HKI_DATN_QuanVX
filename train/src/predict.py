import os
from pathlib import Path
from data_utils import load_models, predict_to_tiff


def main():
    print("🚀 Flood Prediction with Config-based Workflow")
    
    # Set GDAL_DATA environment variable
    gdal_data_path = Path("C:/Users/CuongHM/anaconda3/envs/quan/Library/share/gdal")
    if gdal_data_path.exists():
        os.environ['GDAL_DATA'] = str(gdal_data_path)
        print(f"✅ GDAL_DATA set to: {gdal_data_path}")
    else:
        print("⚠️ GDAL_DATA path not found, continuing anyway...")
    
    # Load models
    model_dir = Path(__file__).parent.parent / "model"
    models = load_models(model_dir)
    
    if not models:
        print("❌ No models found!")
        return
    
    # Run prediction to TIFF
    data_file = Path(r"D:\QuanVX\QuanVX\Default.gdb\a000000c8.gdbtable")  # Update this path for your system
    output_dir = Path(__file__).parent.parent / "results"
    layer_name = "RasterT_Extract1"
    
    predict_to_tiff(
        models=models,
        file_path=data_file, 
        layer_name=layer_name,
        output_dir=output_dir,
        chunk_size=50000  # Giảm từ 50000 xuống 5000 để tránh lỗi memory
    )
    
    print("🎉 Prediction completed!")


if __name__ == "__main__":
    main()