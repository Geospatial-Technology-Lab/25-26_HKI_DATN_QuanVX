"""
Fill null values in TIFF files using neighboring pixel values.
Gradually fills from the boundary inward, constrained by a shapefile boundary.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import geometry_mask
from scipy import ndimage
import geopandas as gpd
import os


def fill_null_values(input_path, output_path, shapefile_path=None, nodata_value=None, max_iterations=1000):
    """
    Fill null/nodata values in a raster using neighboring valid pixels.
    Fills gradually from the edge inward, constrained by shapefile boundary.
    
    Parameters:
    -----------
    input_path : str
        Path to input TIFF file
    output_path : str
        Path to output TIFF file
    shapefile_path : str, optional
        Path to shapefile defining the valid boundary area. 
        Only pixels within this boundary will be filled.
    nodata_value : float, optional
        Value representing nodata. If None, will use the nodata value from the file metadata.
    max_iterations : int
        Maximum number of iterations for filling
    """
    
    # Read the input raster
    with rasterio.open(input_path) as src:
        data = src.read(1)  # Read first band
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        
        # Get nodata value from file if not specified
        if nodata_value is None:
            nodata_value = src.nodata
            if nodata_value is None:
                nodata_value = np.nan
    
    print(f"Input shape: {data.shape}")
    print(f"Nodata value: {nodata_value}")
    
    # Create boundary mask from shapefile if provided
    boundary_mask = None
    if shapefile_path:
        print(f"Loading shapefile boundary: {shapefile_path}")
        try:
            # Read shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Reproject to match raster CRS if needed
            if gdf.crs != crs:
                print(f"Reprojecting shapefile from {gdf.crs} to {crs}")
                gdf = gdf.to_crs(crs)
            
            # Create mask from geometries (True = inside boundary, False = outside)
            boundary_mask = ~geometry_mask(
                gdf.geometry,
                out_shape=data.shape,
                transform=transform,
                invert=False
            )
            
            pixels_in_boundary = np.sum(boundary_mask)
            print(f"Pixels within shapefile boundary: {pixels_in_boundary}")
            
        except Exception as e:
            print(f"Warning: Could not load shapefile: {str(e)}")
            print("Continuing without boundary constraint...")
            boundary_mask = None
    
    # Create mask for pixels that have data (not nodata AND not NaN)
    # Handle both nodata value and NaN
    if np.isnan(nodata_value):
        has_data_mask = ~np.isnan(data)
    else:
        # Check for both nodata value AND NaN
        has_data_mask = (data != nodata_value) & (~np.isnan(data))
    
    print(f"Pixels with data: {np.sum(has_data_mask)}")
    print(f"NaN pixels: {np.sum(np.isnan(data))}")
    if not np.isnan(nodata_value):
        print(f"NoData ({nodata_value}) pixels: {np.sum(data == nodata_value)}")
    
    # Determine which pixels need to be filled based on shapefile
    if boundary_mask is not None:
        # Pixels that SHOULD have data = inside shapefile boundary
        should_have_data = boundary_mask
        
        # Pixels that are NULL = inside boundary BUT don't have data
        null_mask = should_have_data & (~has_data_mask)
        
        # Valid mask = pixels that currently have valid data
        mask = has_data_mask.copy()
        
        print(f"Pixels inside shapefile boundary: {np.sum(should_have_data)}")
        print(f"Pixels with data inside boundary: {np.sum(should_have_data & has_data_mask)}")
        print(f"NULL pixels to fill (in boundary but no data): {np.sum(null_mask)}")
        
        fillable_area = should_have_data
        null_count = np.sum(null_mask)
    else:
        # No boundary constraint - use original logic
        mask = has_data_mask
        fillable_area = np.ones_like(mask, dtype=bool)
        null_mask = ~mask
        null_count = np.sum(null_mask)
        print(f"Null pixels to fill: {null_count}")
    
    if null_count == 0:
        print("No null values to fill!")
        return
    
    # Create a copy of the data to modify
    filled_data = data.copy()
    
    # Define 8-neighbor kernel for checking adjacent pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Iteratively fill null values
    iteration = 0
    filled_count = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Find null pixels that have at least one valid neighbor
        # Dilate the valid mask to find border pixels
        dilated_mask = ndimage.binary_dilation(mask, structure=kernel)
        
        # Border pixels are those that are null but adjacent to valid pixels
        # AND within the fillable area
        border_pixels = dilated_mask & (~mask) & fillable_area
        
        if not np.any(border_pixels):
            print(f"No more border pixels to fill after {iteration} iterations")
            break
        
        # Get indices of border pixels
        border_indices = np.where(border_pixels)
        
        # Fill each border pixel with the mean of its valid neighbors
        for i, j in zip(border_indices[0], border_indices[1]):
            # Get neighborhood
            i_min = max(0, i - 1)
            i_max = min(data.shape[0], i + 2)
            j_min = max(0, j - 1)
            j_max = min(data.shape[1], j + 2)
            
            neighborhood = filled_data[i_min:i_max, j_min:j_max]
            neighborhood_mask = mask[i_min:i_max, j_min:j_max]
            
            # Calculate mean of valid neighbors
            if np.any(neighborhood_mask):
                valid_values = neighborhood[neighborhood_mask]
                filled_data[i, j] = np.mean(valid_values)
                mask[i, j] = True
                filled_count += 1
        
        if iteration % 10 == 0:
            remaining = null_count - filled_count
            print(f"Iteration {iteration}: Filled {filled_count}/{null_count} pixels, {remaining} remaining")
    
    print(f"Filling complete! Total pixels filled: {filled_count}")
    
    # Update profile for output
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=nodata_value if not np.isnan(nodata_value) else None
    )
    
    # Write output
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filled_data.astype(rasterio.float32), 1)
    
    print(f"Output saved to: {output_path}")


def process_directory_inplace(input_dir, shapefile_path=None, nodata_value=None, max_iterations=1000, subfolders=['rf', 'svr', 'xgb'], exclude_folders=['thresholded']):
    """
    Process all TIFF files in specified subdirectories and replace them in place.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing subdirectories with TIFF files
    shapefile_path : str, optional
        Path to shapefile defining the valid boundary area
    nodata_value : float, optional
        Value representing nodata
    max_iterations : int
        Maximum number of iterations for filling
    subfolders : list
        List of subdirectory names to process
    exclude_folders : list
        List of subdirectory names to exclude
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    
    # Count statistics
    total_files = 0
    success_count = 0
    fail_count = 0
    
    # Process each specified subfolder
    for subfolder in subfolders:
        subfolder_path = input_path / subfolder
        
        if not subfolder_path.exists():
            print(f"⚠ Thư mục không tồn tại: {subfolder}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Đang xử lý thư mục: {subfolder}")
        print(f"{'='*60}")
        
        # Find all TIFF files in this subfolder
        tiff_files = list(subfolder_path.glob("*.tif")) + list(subfolder_path.glob("*.tiff"))
        
        for tiff_file in tiff_files:
            # Skip if file is in excluded folders
            if any(excluded in str(tiff_file) for excluded in exclude_folders):
                continue
            
            total_files += 1
            
            print(f"\n[{total_files}] Đang xử lý: {subfolder}/{tiff_file.name}")
            
            # Create temporary output file
            temp_file = tiff_file.parent / f"temp_filled_{tiff_file.name}"
            
            try:
                # Fill null values and save to temp file
                fill_null_values(str(tiff_file), str(temp_file), shapefile_path, nodata_value, max_iterations)
                
                # Replace original file with filled file
                tiff_file.unlink()
                temp_file.rename(tiff_file)
                print(f"  → Đã thay thế file gốc thành công")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Lỗi khi xử lý {tiff_file.name}: {str(e)}")
                # Clean up temp file if exists
                if temp_file.exists():
                    temp_file.unlink()
                fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TỔNG KẾT:")
    print(f"  Tổng số file: {total_files}")
    print(f"  Thành công: {success_count}")
    print(f"  Thất bại: {fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Shapefile path for boundary constraint
    shapefile_boundary = r"C:\Users\Admin\Desktop\GL\gl.shp"
    
    # Danh sách các file TIFF cần xử lý
    input_files = [
        r"D:\prj\feature\gialai_curvature.tif",
        r"D:\prj\feature\gialai_twi.tif",
        r"D:\prj\feature\rainfall_30m.tif"
    ]
    
    print("BẮT ĐẦU FILL GIÁ TRỊ NULL/NaN TRONG ẢNH TIFF")
    print("="*60)
    print(f"Shapefile: {shapefile_boundary}")
    print(f"Số lượng file cần xử lý: {len(input_files)}")
    print("="*60)
    
    # Xử lý từng file
    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Đang xử lý: {os.path.basename(input_file)}")
        print(f"{'='*60}")
        
        # Tạo file tạm thời
        temp_file = input_file.replace('.tif', '_temp.tif')
        
        try:
            # Fill null values
            fill_null_values(
                input_file,
                temp_file,
                shapefile_path=shapefile_boundary,
                nodata_value=None,  # Auto-detect from file
                max_iterations=1000
            )
            
            # Ghi đè file gốc
            if os.path.exists(temp_file):
                os.remove(input_file)
                os.rename(temp_file, input_file)
                print(f"✓ Đã ghi đè file gốc: {os.path.basename(input_file)}")
        
        except Exception as e:
            print(f"✗ Lỗi khi xử lý {os.path.basename(input_file)}: {str(e)}")
            # Xóa file tạm nếu có lỗi
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    print(f"\n{'='*60}")
    print("HOÀN THÀNH XỬ LÝ TẤT CẢ CÁC FILE")
    print(f"{'='*60}")
