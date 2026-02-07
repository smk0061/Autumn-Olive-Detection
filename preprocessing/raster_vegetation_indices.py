# raster_vegetation_indices.py
# calculates 15 vegetation indices from 5-band multispectral rasters
# (blue, green, red, rededge, nir) and stacks them into a 20-band composite.
# exports per-index rasters, the composite, and a summary stats csv.
#
# inputs: 5-band orthomosaic (.tif) with bands ordered blue, green, red, rededge, nir
# outputs: 15 index rasters, 1 composite 20-band raster, 1 stats csv

import os
import csv
import numpy as np
import rasterio

# Directories
input_raster = ""  # 5-band raster (blue, green, red, rededge, nir)
output_folder = ""

os.makedirs(output_folder, exist_ok=True)

# helper functions
def safe_divide(numerator, denominator):
    """element-wise division that returns 0 where denominator is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator != 0, numerator / denominator, 0.0)
    return result.astype(np.float32)


def save_single_band(array, profile, output_path):
    """write a single-band float32 raster."""
    out_profile = profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'lzw'
    })
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)


# Main
def main():
    print(f"processing: {os.path.basename(input_raster)}")

    with rasterio.open(input_raster) as src:
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        rededge = src.read(4).astype(np.float32)
        nir = src.read(5).astype(np.float32)
        profile = src.profile.copy()

    # vegetation indices
    indices = {}

    # normalized difference vegetation index
    indices['NDVI'] = safe_divide(nir - red, nir + red)

    # normalized difference red-edge index
    indices['NDRE'] = safe_divide(nir - rededge, nir + rededge)

    # green normalized difference vegetation index
    indices['GNDVI'] = safe_divide(nir - green, nir + green)

    # blue normalized difference vegetation index
    indices['BNDVI'] = safe_divide(nir - blue, nir + blue)

    # leaf chlorophyll index
    indices['LCI'] = safe_divide(nir - rededge, nir + red)

    # green chlorophyll index
    indices['GCI'] = safe_divide(nir, green) - 1.0

    # red edge chlorophyll index
    indices['RECI'] = safe_divide(nir, rededge) - 1.0

    # simple ratio index
    indices['SRI'] = safe_divide(nir, red)

    # green-red normalized difference vegetation index
    indices['GRNDVI'] = safe_divide(nir - (green + red), nir + (green + red))

    # optimized soil adjusted vegetation index
    indices['OSAVI'] = safe_divide(nir - red, nir + red + 0.16)

    # enhanced vegetation index 2
    indices['EVI2'] = safe_divide(2.5 * (nir - red), nir + 2.4 * red + 1.0)

    # red edge green index
    indices['ReGI'] = safe_divide(rededge, green)

    # green-red vegetation index
    indices['GRVI'] = safe_divide(green - red, green + red)

    # chlorophyll vegetation index
    indices['CVI'] = safe_divide(nir * red, green * green)

    # green-blue vegetation index
    indices['GBVI'] = safe_divide(green - blue, green + blue)

    # save individual index rasters
    for name, array in indices.items():
        out_path = os.path.join(output_folder, f"{name}.tif")
        save_single_band(array, profile, out_path)
        print(f"    saved {name}.tif")

    # stack all 20 bands (5 original + 15 indices) into composite
    input_name = os.path.splitext(os.path.basename(input_raster))[0]
    composite_path = os.path.join(output_folder, f"{input_name}_20band.tif")

    band_arrays = [blue, green, red, rededge, nir] + [indices[k] for k in indices]

    composite_profile = profile.copy()
    composite_profile.update({
        'count': len(band_arrays),
        'dtype': 'float32',
        'compress': 'lzw'
    })

    with rasterio.open(composite_path, 'w', **composite_profile) as dst:
        for i, band in enumerate(band_arrays, start=1):
            dst.write(band.astype(np.float32), i)

    print(f"    saved 20-band composite: {input_name}_20band.tif")

    # calculate stats for individual index rasters only (skip composite)
    stats_data = []
    for name, array in indices.items():
        valid = array[np.isfinite(array)]
        if valid.size > 0:
            stats_data.append({
                'Raster': name,
                'Min': float(np.min(valid)),
                'Max': float(np.max(valid)),
                'Mean': float(np.mean(valid)),
                'StdDev': float(np.std(valid))
            })

    # include original bands in stats
    for band_name, array in zip(['Blue', 'Green', 'Red', 'RedEdge', 'NIR'],
                                 [blue, green, red, rededge, nir]):
        valid = array[np.isfinite(array)]
        if valid.size > 0:
            stats_data.append({
                'Raster': band_name,
                'Min': float(np.min(valid)),
                'Max': float(np.max(valid)),
                'Mean': float(np.mean(valid)),
                'StdDev': float(np.std(valid))
            })

    stats_csv = os.path.join(output_folder, f"{input_name}_raster_statistics.csv")
    with open(stats_csv, 'w', newline='') as csvfile:
        fieldnames = ['Raster', 'Min', 'Max', 'Mean', 'StdDev']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_data)

    print(f"    saved stats: {input_name}_raster_statistics.csv")


if __name__ == '__main__':
    main()