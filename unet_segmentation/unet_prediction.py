# unet_prediction.py
# applies a trained u-net model to full orthomosaics for semantic segmentation.
# processes large images via tiled inference with overlap blending to avoid
# edge artifacts. outputs a class prediction map, confidence map, and color
# visualization, all georeferenced.
#
# supports rgb (3-band), multispectral (5-band), and vi-composite (8-band).
#
# inputs: trained model checkpoint (.pth), orthomosaic (.tif)
# outputs: prediction raster, confidence raster, color visualization raster, log

import numpy as np
import os
import rasterio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_model import UNet, get_input_channels


# configuration - set these before running

MODEL_PATH = ""     # path to trained model checkpoint (.pth)
INPUT_IMAGE = ""    # path to input orthomosaic
OUTPUT_DIR = ""
MODEL_TYPE = ""     # rgb, multispectral, vi_composite

DEVICE_ID = 0       # 0 for first gpu, -1 for cpu
TILE_SIZE = 1024    # adjust based on gpu memory
OVERLAP = 128       # overlap for tile blending

CLASS_LABELS = ["Background", "Barren", "LowVeg", "AutumnOlive", "OtherTree"]
CLASS_COLORS = [
    (0, 0, 0),       # background - black
    (128, 128, 128),  # barren - gray
    (0, 255, 0),      # lowveg - green
    (255, 165, 0),    # autumnolive - orange
    (0, 0, 255)       # othertree - blue
]


# helper functions

def validate_image_channels(img, model_type):
    expected_channels = get_input_channels(model_type)

    if img.shape[0] != expected_channels:
        print(f"WARNING: image has {img.shape[0]} channels, expected {expected_channels} for {model_type}")
        if img.shape[0] > expected_channels:
            print(f"using only the first {expected_channels} channels")
            return img[:expected_channels]
        else:
            print(f"ERROR: image has fewer than {expected_channels} channels, cannot proceed")
            return None

    return img


def create_color_map(prediction, class_colors=CLASS_COLORS):
    height, width = prediction.shape
    color_map = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in enumerate(class_colors):
        mask = prediction == class_idx
        color_map[mask] = color

    return color_map


def pad_to_multiple(image, multiple=16):
    _, h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return padded, pad_h, pad_w


def predict_tile(model, tile, device):
    padded_tile, pad_h, pad_w = pad_to_multiple(tile, 16)

    tile_tensor = torch.from_numpy(padded_tile).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tile_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, dim=1)

    preds = preds[0].cpu().numpy()
    confidence = confidence[0].cpu().numpy()

    if pad_h > 0:
        preds = preds[:-pad_h, :]
        confidence = confidence[:-pad_h, :]
    if pad_w > 0:
        preds = preds[:, :-pad_w]
        confidence = confidence[:, :-pad_w]

    return preds, confidence


def make_blending_weights(tile_h, tile_w, overlap, y_start, x_start, y_end, x_end, img_h, img_w):
    """create 2d blending weight map that fades at overlapping edges.
    uses multiplicative combination so corners fade in both directions."""
    weight_y = np.ones(tile_h, dtype=np.float32)
    weight_x = np.ones(tile_w, dtype=np.float32)

    # fade top edge
    if y_start > 0:
        for k in range(min(overlap, tile_h)):
            weight_y[k] = k / overlap

    # fade bottom edge
    if y_end < img_h:
        for k in range(min(overlap, tile_h)):
            weight_y[tile_h - k - 1] = min(weight_y[tile_h - k - 1], k / overlap)

    # fade left edge
    if x_start > 0:
        for k in range(min(overlap, tile_w)):
            weight_x[k] = k / overlap

    # fade right edge
    if x_end < img_w:
        for k in range(min(overlap, tile_w)):
            weight_x[tile_w - k - 1] = min(weight_x[tile_w - k - 1], k / overlap)

    # outer product gives 2d weights with proper corner fading
    return weight_y[:, None] * weight_x[None, :]


def predict_with_tiles(model, image, device, tile_size=1024, overlap=64):
    _, h, w = image.shape

    # accumulate weighted confidence per class, then pick the winner
    num_classes = len(CLASS_LABELS)
    class_conf = np.zeros((num_classes, h, w), dtype=np.float32)

    step = tile_size - overlap

    n_tiles_h = int(np.ceil(h / step))
    n_tiles_w = int(np.ceil(w / step))
    total_tiles = n_tiles_h * n_tiles_w

    print(f"processing {total_tiles} tiles ({n_tiles_h}x{n_tiles_w}) with size {tile_size}x{tile_size} and {overlap}px overlap")

    tile_count = 0
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            tile_count += 1

            y_start = min(i * step, h - tile_size)
            x_start = min(j * step, w - tile_size)
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            tile_h = y_end - y_start
            tile_w = x_end - x_start

            tile = image[:, y_start:y_end, x_start:x_end]

            if tile_count % 50 == 0 or tile_count == total_tiles:
                print(f"  tile {tile_count}/{total_tiles}")

            try:
                preds, conf = predict_tile(model, tile, device)

                tile_weight = make_blending_weights(tile_h, tile_w, overlap,
                                                     y_start, x_start, y_end, x_end, h, w)

                # accumulate weighted confidence per predicted class
                for cls in range(num_classes):
                    cls_mask = (preds == cls).astype(np.float32)
                    class_conf[cls, y_start:y_end, x_start:x_end] += cls_mask * conf * tile_weight

            except Exception as e:
                print(f"error processing tile {tile_count}: {e}")
                continue

    # final prediction is the class with highest accumulated weighted confidence
    pred = np.argmax(class_conf, axis=0).astype(np.uint8)
    conf = np.max(class_conf, axis=0)

    return pred, conf


# model loading

def load_model(model_path, device, model_type):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        expected_channels = get_input_channels(model_type)
        model_config = checkpoint.get('model_config', {
            'in_channels': expected_channels,
            'num_filters': 64,
            'kernel_size': 3,
            'num_classes': 5,
            'dropout_rate': 0.3
        })

        if model_config['in_channels'] != expected_channels:
            print(f"WARNING: model expects {model_config['in_channels']} channels but {model_type} requires {expected_channels}")

        # dropout set to 0 for inference
        model = UNet(
            in_channels=model_config['in_channels'],
            num_filters=model_config['num_filters'],
            kernel_size=model_config['kernel_size'],
            activation=nn.ReLU,
            num_classes=model_config['num_classes'],
            dropout_rate=0.0
        ).to(device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"model loaded from {model_path}")
        print(f"  config: {model_config}")
        return model
    except Exception as e:
        print(f"error loading model: {e}")
        raise


# main prediction

def run_prediction():
    start_time = time.time()

    if MODEL_TYPE not in ['rgb', 'multispectral', 'vi_composite']:
        raise ValueError(f"invalid MODEL_TYPE: {MODEL_TYPE}. must be 'rgb', 'multispectral', or 'vi_composite'")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if DEVICE_ID >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{DEVICE_ID}")
        print(f"using gpu: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("using cpu")

    if not os.path.exists(INPUT_IMAGE):
        print(f"ERROR: input image not found: {INPUT_IMAGE}")
        return

    print(f"processing: {INPUT_IMAGE}")
    print(f"model type: {MODEL_TYPE}")

    try:
        with rasterio.open(INPUT_IMAGE) as src:
            img = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs
            height, width = img.shape[1], img.shape[2]

        print(f"input shape: {img.shape} ({width}x{height} pixels)")

        img = validate_image_channels(img, MODEL_TYPE)
        if img is None:
            return

        if np.isnan(img).any() or np.isinf(img).any():
            nan_count = np.isnan(img).sum()
            inf_count = np.isinf(img).sum()
            print(f"replacing {nan_count} nan and {inf_count} inf values")
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"image stats - min: {img.min():.4f}, max: {img.max():.4f}, mean: {img.mean():.4f}")

        basename = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
        pred_filename = os.path.join(OUTPUT_DIR, f"{basename}_pred.tif")
        conf_filename = os.path.join(OUTPUT_DIR, f"{basename}_conf.tif")
        color_filename = os.path.join(OUTPUT_DIR, f"{basename}_color.tif")

        model = load_model(MODEL_PATH, device, MODEL_TYPE)
        model.eval()

        pred, conf = predict_with_tiles(model, img, device, tile_size=TILE_SIZE, overlap=OVERLAP)

        color_map = create_color_map(pred, CLASS_COLORS)

        # class distribution
        class_counts = {}
        for i in range(len(CLASS_LABELS)):
            count = np.sum(pred == i)
            percentage = (count / pred.size) * 100
            class_counts[CLASS_LABELS[i]] = (count, percentage)

        print("\nclass distribution:")
        for class_name, (count, percentage) in class_counts.items():
            print(f"  {class_name}: {count} pixels ({percentage:.2f}%)")

        # save prediction
        with rasterio.open(pred_filename, 'w', driver='GTiff',
                           height=height, width=width, count=1,
                           dtype=np.uint8, crs=crs, transform=transform) as dst:
            dst.write(pred.astype(np.uint8), 1)

        # save confidence
        with rasterio.open(conf_filename, 'w', driver='GTiff',
                           height=height, width=width, count=1,
                           dtype=np.float32, crs=crs, transform=transform) as dst:
            dst.write(conf.astype(np.float32), 1)

        # save color visualization
        with rasterio.open(color_filename, 'w', driver='GTiff',
                           height=height, width=width, count=3,
                           dtype=np.uint8, crs=crs, transform=transform) as dst:
            for i in range(3):
                dst.write(color_map[:, :, i], i + 1)

        elapsed = time.time() - start_time
        print(f"\nprediction completed in {elapsed:.2f} seconds")

        # save log
        log_file = os.path.join(OUTPUT_DIR, f"{basename}_prediction_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"input image: {INPUT_IMAGE}\n")
            f.write(f"model: {MODEL_PATH}\n")
            f.write(f"model type: {MODEL_TYPE}\n")
            f.write(f"device: {device}\n")
            f.write(f"image size: {width}x{height} pixels\n")
            f.write(f"processing time: {elapsed:.2f} seconds\n\n")

            f.write(f"class distribution:\n")
            for class_name, (count, percentage) in class_counts.items():
                f.write(f"  {class_name}: {count} pixels ({percentage:.2f}%)\n")

            f.write(f"\noutput files:\n")
            f.write(f"  prediction: {pred_filename}\n")
            f.write(f"  confidence: {conf_filename}\n")
            f.write(f"  color map: {color_filename}\n")

        print(f"log saved to: {log_file}")

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"CUDA out of memory: {e}")
            print("try reducing TILE_SIZE or set DEVICE_ID=-1 for cpu")
        else:
            print(f"error: {e}")
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_prediction()
