import os
from pathlib import Path
from data_utils import load_models, predict_to_tiff


def main():
    print("🚀 Flood Prediction")
    
    # Set GDAL_DATA
    gdal_data_path = Path("C:/Users/CuongHM/anaconda3/envs/quan/Library/share/gdal")
    if gdal_data_path.exists():
        os.environ['GDAL_DATA'] = str(gdal_data_path)
        print(f"✅ GDAL_DATA set")
    
    # Load models
    model_dir = Path(__file__).parent.parent / "model"
    models = load_models(model_dir)
    if not models:
        print("❌ No models found!"); return
    
    # Data file
    data_file = Path(r"D:\QuanVX\QuanVX\Default.gdb\a000000c8.gdbtable")
    output_dir = Path(__file__).parent.parent / "results"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}"); return
    
    predict_to_tiff(
        models=models,
        file_path=data_file, 
        layer_name="RasterT_Extract1",
        output_dir=output_dir,
        chunk_size=50000
    )
    
    print("🎉 Done!")


if __name__ == "__main__":
    main()