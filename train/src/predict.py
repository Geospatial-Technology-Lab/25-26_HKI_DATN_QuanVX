import os
from pathlib import Path
from data_utils import predict_to_tiff


def main():
    # Set GDAL_DATA
    gdal_paths = ["/usr/share/gdal", "/usr/local/share/gdal"]
    for gdal_path in gdal_paths:
        if Path(gdal_path).exists():
            os.environ['GDAL_DATA'] = str(gdal_path)
            break
    
    # Data file path - update this
    data_file = Path(r"D:\QuanVX\QuanVX\Default.gdb\a000000c8.gdbtable")
    if not data_file.exists():
        print("Data file not found! Update the path in predict.py")
        return
    
    output_dir = Path(__file__).parent.parent / "results"
    
    predict_to_tiff(
        file_path=data_file, 
        layer_name="RasterT_Extract1",
        output_dir=output_dir,
        chunk_size=50000
    )


if __name__ == "__main__":
    main()