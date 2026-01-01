"""
Debug script to check TIFF file and shapefile information
"""

import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd

# Paths
tiff_file = r"D:\prj\results\map\cliped\rf\cliped_flood_probability_pso_RF.tif"
shapefile = r"C:\Users\Admin\Desktop\GL\gl.shp"

print("="*60)
print("CHECKING TIFF FILE")
print("="*60)

# Read TIFF
with rasterio.open(tiff_file) as src:
    data = src.read(1)
    profile = src.profile
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    
    print(f"Shape: {data.shape}")
    print(f"CRS: {crs}")
    print(f"Transform: {transform}")
    print(f"NoData value: {nodata}")
    print(f"Data type: {data.dtype}")
    print(f"Data min: {np.nanmin(data)}")
    print(f"Data max: {np.nanmax(data)}")
    print(f"Data mean: {np.nanmean(data)}")
    
    # Check unique values (sample)
    unique_vals = np.unique(data.flatten())
    print(f"\nNumber of unique values: {len(unique_vals)}")
    print(f"First 10 unique values: {unique_vals[:10]}")
    print(f"Last 10 unique values: {unique_vals[-10:]}")
    
    # Check for NaN
    nan_count = np.sum(np.isnan(data))
    print(f"\nNaN pixels: {nan_count}")
    
    # Check for nodata value
    if nodata is not None:
        nodata_count = np.sum(data == nodata)
        print(f"NoData ({nodata}) pixels: {nodata_count}")
    
    # Check for zero values
    zero_count = np.sum(data == 0)
    print(f"Zero pixels: {zero_count}")
    
    # Check for negative values
    neg_count = np.sum(data < 0)
    print(f"Negative pixels: {neg_count}")

print("\n" + "="*60)
print("CHECKING SHAPEFILE")
print("="*60)

# Read shapefile
gdf = gpd.read_file(shapefile)
print(f"Shapefile CRS: {gdf.crs}")
print(f"Number of features: {len(gdf)}")
print(f"Geometry type: {gdf.geometry.type.unique()}")
print(f"Bounds: {gdf.total_bounds}")

# Reproject if needed
if gdf.crs != crs:
    print(f"\nReprojecting shapefile from {gdf.crs} to {crs}")
    gdf = gdf.to_crs(crs)
    print(f"New bounds: {gdf.total_bounds}")

print("\n" + "="*60)
print("CREATING BOUNDARY MASK")
print("="*60)

# Create boundary mask
with rasterio.open(tiff_file) as src:
    boundary_mask = ~geometry_mask(
        gdf.geometry,
        out_shape=data.shape,
        transform=transform,
        invert=False
    )
    
    print(f"Boundary mask shape: {boundary_mask.shape}")
    print(f"Pixels inside boundary: {np.sum(boundary_mask)}")
    print(f"Pixels outside boundary: {np.sum(~boundary_mask)}")

print("\n" + "="*60)
print("ANALYZING DATA VS BOUNDARY")
print("="*60)

# Check data inside/outside boundary
if nodata is not None:
    has_data = (data != nodata) & (~np.isnan(data))  # Check BOTH nodata and NaN
else:
    has_data = ~np.isnan(data)

print(f"Total pixels with data: {np.sum(has_data)}")
print(f"Pixels with data INSIDE boundary: {np.sum(has_data & boundary_mask)}")
print(f"Pixels with data OUTSIDE boundary: {np.sum(has_data & ~boundary_mask)}")
print(f"Pixels WITHOUT data INSIDE boundary: {np.sum(~has_data & boundary_mask)}")
print(f"Pixels WITHOUT data OUTSIDE boundary: {np.sum(~has_data & ~boundary_mask)}")

# Find edges where data might be missing
print("\n" + "="*60)
print("CHECKING EDGE PIXELS")
print("="*60)

# Check corners and edges
h, w = data.shape
corners = [
    (0, 0), (0, w-1), (h-1, 0), (h-1, w-1),  # corners
    (h//4, w//4), (h//4, 3*w//4), (3*h//4, w//4), (3*h//4, 3*w//4)  # mid points
]

for i, j in corners:
    val = data[i, j]
    in_boundary = boundary_mask[i, j]
    print(f"Pixel [{i},{j}]: value={val:.6f}, in_boundary={in_boundary}, is_nodata={val==nodata if nodata else np.isnan(val)}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

null_pixels = np.sum(~has_data & boundary_mask)
if null_pixels == 0:
    print("⚠ No null pixels found inside boundary!")
    print("\nPossible reasons:")
    print("1. All pixels inside boundary already have valid data")
    print("2. NoData value detection is incorrect")
    print("3. Shapefile boundary doesn't match expected area")
    print("\nTry checking:")
    print(f"- If nodata={nodata} is correct")
    print("- If shapefile covers the expected area")
    print("- Visual inspection of the TIFF file")
else:
    print(f"✓ Found {null_pixels} null pixels to fill")
