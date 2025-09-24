import os
from pathlib import Path
from data_utils import predict_to_tiff


def main():
    
    data_file = Path(r"Z:\guest01\QuanVX\QuanVX\QuanVX\Default.gdb")
    
    output_dir = Path(__file__).parent.parent / "results"
    
    predict_to_tiff(
        file_path=data_file, 
        layer_name="RasterT_Extract1",
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()