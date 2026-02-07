# image_chipper.py
# creates training chips from orthomosaics and annotation shapefiles for
# u-net semantic segmentation. slides a window across each orthomosaic,
# rasterizes intersecting annotations into class masks, and exports
# image/mask pairs that meet a minimum annotation coverage threshold.
#
# inputs: orthomosaics (.tif) and annotation shapefiles (.shp)
# outputs: 256x256 image chips and corresponding class mask chips

import os
import re
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
import geopandas as gpd
import rasterio.features
from shapely.geometry import box
import concurrent.futures
from tqdm import tqdm

# directories
ortho_folder = ""
annotation_parent_folder = ""
output_images = ""
output_masks = ""

# parameters
chip_size = 256
stride = chip_size // 2  # 50% overlap
min_coverage = 0.1       # 10% annotation coverage threshold

# site name mapping for matching orthos to annotation folders
site_name_mapping = {
    "LittleIndianCreek": "LIC"
}

# class mapping
class_mapping = {
    "Barren": 1,
    "LowVeg": 2,
    "AutumnOlive": 3,
    "UnknownTree": 4
}

# processing functions
def process_chip(image_path, window, annotations_gdf, chip_name, out_img_folder, out_mask_folder, min_coverage):
    """extract a single image chip and its corresponding annotation mask."""
    try:
        with rasterio.open(image_path) as src:
            chip_img = src.read(window=window)
            chip_transform = rasterio.windows.transform(window, src.transform)
            chip_bounds = window_bounds(window, src.transform)
            chip_polygon = box(*chip_bounds)
            img_profile = src.profile.copy()

        # select annotations that intersect the chip
        intersecting = annotations_gdf[annotations_gdf.geometry.intersects(chip_polygon)]
        if intersecting.empty:
            return None

        # convert annotations to rasterization shapes
        # use ClassCode (1-4) if valid, otherwise fall back to ClassType mapping
        shapes = []
        for geom, val, typ in zip(intersecting.geometry, intersecting["ClassCode"], intersecting["ClassType"]):
            if not np.isfinite(val):
                int_val = 0
            else:
                try:
                    int_val = int(val)
                except Exception:
                    int_val = 0
            if int_val not in (1, 2, 3, 4):
                int_val = class_mapping.get(typ, 0)
            shapes.append((geom, int_val))

        # rasterize annotations into mask
        mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(chip_img.shape[1], chip_img.shape[2]),
            transform=chip_transform,
            fill=0,
            dtype='uint8'
        )

        # skip chips below minimum annotation coverage
        total_pixels = mask.size
        annotated_pixels = np.count_nonzero(mask)
        coverage = annotated_pixels / float(total_pixels)
        if coverage < min_coverage:
            return None

        out_img_path = os.path.join(out_img_folder, chip_name)
        out_mask_path = os.path.join(out_mask_folder, chip_name.replace('.tif', '_mask.tif'))

        # write image chip
        img_profile.update({
            'height': chip_img.shape[1],
            'width': chip_img.shape[2],
            'transform': chip_transform
        })

        with rasterio.open(out_img_path, 'w', **img_profile) as dst:
            dst.write(chip_img)

        # write mask chip as uint8
        mask_profile = {
            'driver': 'GTiff',
            'height': chip_img.shape[1],
            'width': chip_img.shape[2],
            'count': 1,
            'dtype': 'uint8',
            'transform': chip_transform,
            'crs': img_profile['crs'],
            'compress': 'lzw',
            'nodata': None
        }

        with rasterio.open(out_mask_path, 'w', **mask_profile) as dst:
            dst.write(mask.astype('uint8'), 1)

        return chip_name
    except Exception as e:
        return f"error processing chip {chip_name}: {e}"


def process_image(image_path, annotation_path, out_img_folder, out_mask_folder, min_coverage, chip_size, stride):
    """generate all chips for a single orthomosaic/annotation pair."""
    annotations_gdf = gpd.read_file(annotation_path)

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        raster_crs = src.crs
    annotations_gdf = annotations_gdf.to_crs(raster_crs)

    # build task list of chip windows
    tasks = []
    base = os.path.splitext(os.path.basename(image_path))[0]

    # simplify filename if it contains '8band'
    parts = base.split("_")
    if len(parts) >= 3 and "8band" in parts[-1]:
        base = f"{parts[0]}_{parts[1]}_{parts[-1]}"

    for row in range(0, height - chip_size + 1, stride):
        for col in range(0, width - chip_size + 1, stride):
            window_obj = Window(col, row, chip_size, chip_size)
            chip_name = f"{base}_{col}-{row}.tif"
            tasks.append((window_obj, chip_name))

    results = []
    error_count = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_chip = {
            executor.submit(process_chip, image_path, window_obj, annotations_gdf, chip_name,
                            out_img_folder, out_mask_folder, min_coverage): chip_name
            for window_obj, chip_name in tasks
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_chip),
                           total=len(future_to_chip),
                           desc="processing chips"):
            try:
                result = future.result()
                if result is not None:
                    if isinstance(result, str) and result.startswith("error processing chip"):
                        error_count += 1
                        print(result)
                    else:
                        results.append(result)
            except Exception as exc:
                error_count += 1
                print(f"unexpected error: {exc}")

    if error_count > 0:
        print(f"encountered {error_count} errors for image {os.path.basename(image_path)}")
    return results


def summarize_outputs(out_img_folder, out_mask_folder):
    """print summary of generated chips and class pixel counts."""
    image_files = sorted([f for f in os.listdir(out_img_folder) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(out_mask_folder) if f.endswith("_mask.tif")])

    print("\n--- summary ---")
    print(f"total image chips: {len(image_files)}")
    print(f"total mask chips: {len(mask_files)}")

    # check image/mask pairing
    image_bases = set(os.path.splitext(f)[0] for f in image_files)
    mask_bases = set(os.path.splitext(f.replace('_mask', ''))[0] for f in mask_files)

    if image_bases == mask_bases:
        print("image chips and mask chips are correctly paired")
    else:
        print("WARNING: mismatch between image and mask chip filenames")

    # aggregate pixel counts across all masks
    pixel_counts = {}
    for mask_file in mask_files:
        mask_path = os.path.join(out_mask_folder, mask_file)
        try:
            with rasterio.open(mask_path) as src:
                arr = src.read(1)
            unique, counts = np.unique(arr, return_counts=True)
            for u, c in zip(unique, counts):
                pixel_counts[u] = pixel_counts.get(u, 0) + c
        except Exception as e:
            print(f"error reading mask {mask_file}: {e}")

    print("pixel counts per class in mask dataset:")
    for cls in sorted(pixel_counts.keys()):
        cls_name = next((name for name, code in class_mapping.items() if code == cls),
                        "Background" if cls == 0 else f"Unknown-{cls}")
        print(f"  class {cls} ({cls_name}): {pixel_counts[cls]} pixels")


# main
def main():
    for folder in [output_images, output_masks]:
        os.makedirs(folder, exist_ok=True)

    # match orthomosaics to annotation shapefiles by site name and date
    pattern = re.compile(r"([A-Za-z]+)_([0-9-]+)")
    ortho_files = [f for f in os.listdir(ortho_folder) if f.endswith(".tif")]
    file_pairs = {}

    for ortho in ortho_files:
        match = pattern.search(ortho)
        if match:
            site, date = match.groups()
            site_key = site_name_mapping.get(site, site)
            annotation_subfolder = os.path.join(annotation_parent_folder, f"{site_key}_{date}_rgb")

            if os.path.exists(annotation_subfolder):
                for shp_file in os.listdir(annotation_subfolder):
                    if shp_file.endswith(".shp") and site_key in shp_file and date in shp_file:
                        file_pairs[os.path.join(ortho_folder, ortho)] = os.path.join(annotation_subfolder, shp_file)
                        break

    print("matched file pairs:")
    for img, shp in file_pairs.items():
        print(f"  {img} -> {shp}")

    proceed = input("press 'y' to proceed: ")
    if proceed.lower() != 'y':
        print("stopping...")
        return

    total_chips = 0
    for image_path, annotation_path in file_pairs.items():
        chips = process_image(image_path, annotation_path, output_images, output_masks, min_coverage, chip_size, stride)
        total_chips += len(chips)
        print(f"processed {len(chips)} chips for {os.path.basename(image_path)}")

    print(f"\ntotal chips processed: {total_chips}")
    summarize_outputs(output_images, output_masks)


if __name__ == '__main__':
    main()
