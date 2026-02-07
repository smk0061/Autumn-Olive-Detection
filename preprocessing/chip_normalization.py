# chip_normalization.py
# normalizes image chips using flight-specific statistics for consistent
# input to u-net training. computes per-flight band statistics with 1st/99th
# percentile clipping, then z-score normalizes each chip against its flight.
# masks are copied unchanged.
#
# supports two data types:
#   rgb: normalizes bands 0-2 (blue, green, red)
#   vi_composite: normalizes bands 5-7 (vegetation indices from 8-band composite)
#
# inputs: chipped image/mask pairs from image_chipper.py
# outputs: normalized image chips, copied masks, flight stats json, processing log

import os
import glob
import re
import numpy as np
import rasterio
import json
import multiprocessing
from tqdm import tqdm


# configuration - set these before running

BASE_DIR = ""
STAGE = ""       # early, peak, late, senescence
DATA_TYPE = ""   # rgb or vi_composite
OUTPUT_DIR = ""
NUM_WORKERS = multiprocessing.cpu_count() - 1


def extract_flight_info(filename):
    """parse site name and date from chip filename."""
    basename = os.path.basename(filename)
    match = re.match(r'([^_]+)_(\d{2}-\d{2}-\d{4})_', basename)
    if match:
        return (match.group(1), match.group(2))
    return None


def get_band_indices(data_type):
    """return which bands to normalize based on data type."""
    if data_type == 'rgb':
        return range(3)       # bands 0, 1, 2
    elif data_type == 'vi_composite':
        return range(5, 8)    # bands 5, 6, 7 (ndvi, ndre, gndvi in 8-band composite)
    else:
        raise ValueError(f"unknown data_type: {data_type}. use 'rgb' or 'vi_composite'")


def get_data_folder(data_type):
    if data_type == 'rgb':
        return 'rgb_chips'
    elif data_type == 'vi_composite':
        return 'vi_comp_chips_corrected'
    else:
        raise ValueError(f"unknown data_type: {data_type}")


def calculate_flight_stats(flight_id, flight_files, band_indices):
    """compute per-band statistics across all chips from a single flight."""
    try:
        all_values = {i: [] for i in band_indices}

        for img_file in flight_files:
            with rasterio.open(img_file) as src:
                img = src.read()

                for i in band_indices:
                    if i >= img.shape[0]:
                        continue

                    band = img[i]
                    finite_values = band[np.isfinite(band)]
                    all_values[i].append(finite_values)

        stats = {}
        for i in band_indices:
            if not all_values[i]:
                continue

            concat_values = np.concatenate(all_values[i])

            if len(concat_values) > 0:
                # percentile clipping to remove extreme outliers
                p1 = np.percentile(concat_values, 1)
                p99 = np.percentile(concat_values, 99)

                clipped = concat_values[(concat_values >= p1) & (concat_values <= p99)]

                # stats computed on clipped data for robust normalization
                stats[i] = {
                    'mean': float(np.mean(clipped)),
                    'std': float(np.std(clipped)),
                    'min': float(np.min(clipped)),
                    'max': float(np.max(clipped)),
                    'p1': float(p1),
                    'p99': float(p99),
                    'num_samples': len(concat_values)
                }

        stats['flight_id'] = flight_id
        stats['num_images'] = len(flight_files)

        return stats

    except Exception as e:
        print(f"error calculating stats for flight {flight_id}: {str(e)}")
        return None


def normalize_image(img_path, flight_stats, band_indices):
    """apply flight-specific z-score normalization to an image chip."""
    with rasterio.open(img_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    if not flight_stats:
        return image, profile

    normalized = image.copy()

    for i in band_indices:
        if i not in flight_stats or i >= normalized.shape[0]:
            continue

        band = normalized[i]
        stats = flight_stats[i]

        # clip to flight-specific percentiles
        band_clipped = np.clip(band, stats['p1'], stats['p99'])

        # z-score normalize using flight-specific mean and std
        std = max(stats['std'], 1e-6)
        normalized[i] = (band_clipped - stats['mean']) / std

    # safety checks for any residual bad values
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, -10.0, 10.0)

    # update profile to float32 since normalized values are no longer in original dtype range
    profile.update({'dtype': 'float32'})

    return normalized, profile


def process_image_wrapper(args):
    """wrapper for multiprocessing image normalization."""
    img_path, output_path, flight_stats, band_indices = args

    try:
        normalized_img, profile = normalize_image(img_path, flight_stats, band_indices)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(normalized_img)

        return True
    except Exception as e:
        print(f"error processing {img_path}: {str(e)}")
        return False


def normalize_dataset():

    data_folder = get_data_folder(DATA_TYPE)
    input_base_dir = os.path.join(BASE_DIR, data_folder, STAGE)

    if not OUTPUT_DIR:
        output_base_dir = os.path.join(BASE_DIR, f"{data_folder}_normalized", STAGE)
    else:
        output_base_dir = OUTPUT_DIR

    input_img_dir = os.path.join(input_base_dir, "imgs")
    input_mask_dir = os.path.join(input_base_dir, "masks")
    output_img_dir = os.path.join(output_base_dir, "imgs")
    output_mask_dir = os.path.join(output_base_dir, "masks")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(input_img_dir, "*.tif")))
    mask_files = sorted(glob.glob(os.path.join(input_mask_dir, "*.tif")))

    if not img_files:
        raise ValueError(f"no image files found in {input_img_dir}")

    print(f"found {len(img_files)} image files")
    print(f"found {len(mask_files)} mask files")

    band_indices = get_band_indices(DATA_TYPE)
    print(f"normalizing bands: {list(band_indices)}")

    # group chips by flight
    print("\ncalculating flight statistics...")
    flight_groups = {}
    for img_file in img_files:
        flight_info = extract_flight_info(img_file)
        if flight_info:
            flight_id = f"{flight_info[0]}_{flight_info[1]}"
            if flight_id not in flight_groups:
                flight_groups[flight_id] = []
            flight_groups[flight_id].append(img_file)

    print(f"found {len(flight_groups)} unique flights:")
    for flight_id, files in flight_groups.items():
        print(f"  {flight_id}: {len(files)} images")

    # compute per-flight statistics
    flight_stats = {}
    for flight_id, flight_files in tqdm(flight_groups.items(), desc="calculating flight stats"):
        stats = calculate_flight_stats(flight_id, flight_files, band_indices)
        if stats:
            flight_stats[flight_id] = stats

    print(f"\ncalculated statistics for {len(flight_stats)} flights")

    # save flight statistics for reference
    stats_file = os.path.join(output_base_dir, f"flight_statistics_{DATA_TYPE}_{STAGE}.json")
    with open(stats_file, 'w') as f:
        json.dump(flight_stats, f, indent=2)
    print(f"saved flight statistics to: {stats_file}")

    # normalize images
    print(f"\nnormalizing {len(img_files)} images...")

    process_args = []
    for img_file in img_files:
        flight_info = extract_flight_info(img_file)
        if flight_info:
            flight_id = f"{flight_info[0]}_{flight_info[1]}"
            img_flight_stats = flight_stats.get(flight_id, {})
        else:
            img_flight_stats = {}

        img_filename = os.path.basename(img_file)
        output_img_path = os.path.join(output_img_dir, img_filename)

        process_args.append((img_file, output_img_path, img_flight_stats, band_indices))

    successful_images = 0
    if NUM_WORKERS == 1:
        for args in tqdm(process_args, desc="normalizing images"):
            if process_image_wrapper(args):
                successful_images += 1
    else:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(process_image_wrapper, process_args),
                total=len(process_args),
                desc="normalizing images"
            ))
            successful_images = sum(results)

    print(f"successfully normalized {successful_images} out of {len(img_files)} images")

    # copy mask files unchanged
    print(f"\ncopying {len(mask_files)} mask files...")
    for mask_file in tqdm(mask_files, desc="copying masks"):
        mask_filename = os.path.basename(mask_file)
        output_mask_path = os.path.join(output_mask_dir, mask_filename)

        with rasterio.open(mask_file) as src:
            mask_data = src.read()
            profile = src.profile.copy()

        with rasterio.open(output_mask_path, 'w', **profile) as dst:
            dst.write(mask_data)

    # processing log
    log_file = os.path.join(output_base_dir, f"normalization_log_{DATA_TYPE}_{STAGE}.txt")
    with open(log_file, 'w') as f:
        f.write(f"dataset normalization complete\n")
        f.write(f"data type: {DATA_TYPE}\n")
        f.write(f"stage: {STAGE}\n")
        f.write(f"input directory: {input_base_dir}\n")
        f.write(f"output directory: {output_base_dir}\n")
        f.write(f"bands normalized: {list(band_indices)}\n")
        f.write(f"total images: {len(img_files)}\n")
        f.write(f"successfully normalized: {successful_images}\n")
        f.write(f"total masks copied: {len(mask_files)}\n")
        f.write(f"flights processed: {len(flight_stats)}\n")

        f.write(f"\nflight statistics summary:\n")
        for flight_id, stats in flight_stats.items():
            f.write(f"  {flight_id}: {stats['num_images']} images\n")

    print(f"\nnormalization complete")
    print(f"normalized dataset saved to: {output_base_dir}")
    print(f"processing log saved to: {log_file}")

    return output_base_dir


if __name__ == '__main__':
    normalize_dataset()