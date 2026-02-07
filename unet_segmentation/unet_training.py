# unet_training.py
# u-net semantic segmentation for invasive species detection from uas imagery.
# supports rgb (3-band), multispectral (5-band), and vi-composite (8-band) inputs.
# includes weighted cross-entropy with focal loss, stratified splitting,
# dynamic class weighting, data augmentation, and multi-gpu support.
#
# preprocessing: run chip_normalization.py first for rgb and vi_composite datasets.
# multispectral data is radiometrically corrected by the sentera 6x sensor.
#
# inputs: image/mask chip pairs from image_chipper.py
# outputs: trained model (.pth), training metrics, confusion matrix, classification report

import albumentations as A
import glob
import numpy as np
import os
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torch.nn.parallel as parallel
from unet_model import UNet, get_input_channels, init_weights


# configuration - set these before running

base_dir = ""
stage = ""        # early, peak, late, senescence
model_type = ""   # rgb, multispectral, vi_composite
output_dir = ""

# training parameters
batch_size = 48
epochs = 200
initial_lr = 0.001
max_lr = 0.005
weight_decay = 1e-4
patience = 15
augment = True

# model architecture parameters
num_filters = 64
kernel_size = 3
num_classes = 5
dropout_rate = 0.3

# loss function parameters
class_priorities = {1: 1.0, 2: 1.0, 3: 1.8, 4: 1.5}
focal_gamma = 2
ce_weight = 1.0
focal_weight = 1.0

# hardware
num_workers = 8


# augmentation

def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.RandomRotate90(p=0.25)
    ], additional_targets={'mask': 'mask'})


# loss function: weighted cross-entropy with focal modulation

class WeightedCrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=0, ce_weight=1.0, focal_weight=1.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            alpha = [min(max(w, 0.1), 10.0) for w in alpha]
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        if torch.isnan(inputs).any():
            inputs = torch.nan_to_num(inputs)

        inputs = inputs.float()
        targets = targets.long()

        ce_loss = F.cross_entropy(inputs, targets,
                                  weight=self.alpha.to(inputs.device) if self.alpha is not None else None,
                                  ignore_index=self.ignore_index, reduction='none')

        with torch.no_grad():
            pt = torch.exp(-ce_loss)
            pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
            focal_mod = (1 - pt) ** self.gamma

        loss = self.ce_weight * ce_loss + self.focal_weight * focal_mod * ce_loss

        if torch.isnan(loss).any():
            non_nan = ~torch.isnan(loss)
            if non_nan.any():
                if self.reduction == 'mean':
                    return loss[non_nan].mean()
                elif self.reduction == 'sum':
                    return loss[non_nan].sum()
                else:
                    return loss * non_nan.float()
            else:
                return torch.tensor(0.1, device=loss.device, requires_grad=True)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# dataset

class UNetSegmentationDataset(Dataset):
    def __init__(self, base_dir, stage, model_type, transform=None, mask_transform=None,
                 augment=False, debug_subset=False, subset_size=100):

        if model_type == 'rgb':
            normalized_dir = os.path.join(base_dir, "rgb_chips_normalized", stage)
            original_dir = os.path.join(base_dir, "rgb_chips", stage)
            if os.path.exists(normalized_dir):
                self.img_dir = os.path.join(normalized_dir, "imgs")
                self.mask_dir = os.path.join(normalized_dir, "masks")
                print("using normalized rgb dataset")
            else:
                self.img_dir = os.path.join(original_dir, "imgs")
                self.mask_dir = os.path.join(original_dir, "masks")
                print("warning: using original rgb dataset (not normalized)")

        elif model_type == 'multispectral':
            self.img_dir = os.path.join(base_dir, "multispec_chips", stage, "imgs")
            self.mask_dir = os.path.join(base_dir, "multispec_chips", stage, "masks")

        elif model_type == 'vi_composite':
            normalized_dir = os.path.join(base_dir, "vi_comp_chips_corrected_normalized", stage)
            original_dir = os.path.join(base_dir, "vi_comp_chips_corrected", stage)
            if os.path.exists(normalized_dir):
                self.img_dir = os.path.join(normalized_dir, "imgs")
                self.mask_dir = os.path.join(normalized_dir, "masks")
                print("using normalized vi-composite dataset")
            else:
                self.img_dir = os.path.join(original_dir, "imgs")
                self.mask_dir = os.path.join(original_dir, "masks")
                print("warning: using original vi-composite dataset (not normalized)")
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.tif")))
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_dir, "*.tif")))
        self.model_type = model_type

        if len(self.img_files) != len(self.mask_files):
            raise ValueError("the number of images and masks do not match")

        if debug_subset:
            self.img_files = self.img_files[:subset_size]
            self.mask_files = self.mask_files[:subset_size]

        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment

        if self.augment:
            self.augmentation = get_train_augmentations()
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = self.mask_files[idx]

        with rasterio.open(img_file) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(mask_file) as src:
            mask = src.read(1).astype(np.uint8)

        if self.augmentation is not None:
            img_np = np.transpose(img, (1, 2, 0))
            augmented = self.augmentation(image=img_np, mask=mask)
            img_np = augmented['image']
            mask = augmented['mask']
            img = np.transpose(img_np, (2, 0, 1))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(mask.astype(np.int64))

        return img, mask


# stratified split using composite class presence labels

def compute_composite_label(mask_path, ignore_index=0):
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int64)
    label = []
    for cls in range(1, 5):
        label.append(1 if (mask == cls).any() else 0)
    return "_".join(map(str, label))


def stratified_split(dataset):
    composite_labels = [compute_composite_label(fp) for fp in dataset.mask_files]
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=composite_labels, random_state=42)
    return train_idx, val_idx


# dynamic class weighting

def calculate_class_distribution(dataset, indices):
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for idx in indices:
        try:
            mask_path = dataset.mask_files[idx]
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int64)
                for cls in range(5):
                    class_counts[cls] += np.sum(mask == cls)
        except Exception:
            continue

    return class_counts


def compute_class_weights(class_counts, smoothing_factor=0.2):
    total_pixels = sum(class_counts.values())
    if total_pixels == 0:
        return [1.0, 1.0, 1.0, 1.0, 1.0]

    frequencies = {cls: count / total_pixels for cls, count in class_counts.items()}

    weights = {}
    for cls, freq in frequencies.items():
        if freq > 0:
            weights[cls] = 1.0 / (freq ** (1.0 - smoothing_factor))
        else:
            weights[cls] = 1.0

    weight_sum = sum(weights.values())
    normalized_weights = {cls: weight / weight_sum * len(class_counts)
                         for cls, weight in weights.items()}

    weight_list = [normalized_weights[i] for i in range(len(class_counts))]
    return weight_list


# architecture imported from unet_model.py


# evaluation

def compute_metrics(pred, target, num_classes=5, ignore_index=0):
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'per_class': {}}

    accuracy = (pred == target).sum().item() / pred.numel()
    per_class = {}
    prec_list, rec_list, f1_list = [], [], []

    for cls in range(1, num_classes):
        tp = ((pred == cls) & (target == cls)).sum().item()
        fp = ((pred == cls) & (target != cls)).sum().item()
        fn = ((pred != cls) & (target == cls)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[cls] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': tp + fn}
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    return {
        'accuracy': accuracy,
        'precision': np.mean(prec_list),
        'recall': np.mean(rec_list),
        'f1': np.mean(f1_list),
        'per_class': per_class
    }


def compute_confusion(pred, target, num_classes=5, ignore_index=0):
    valid = target != ignore_index
    pred = pred[valid].cpu().numpy()
    target = target[valid].cpu().numpy()
    return confusion_matrix(target, pred, labels=list(range(1, num_classes)))


def evaluate_model(model, dataloader, device, criterion, num_classes=5, ignore_index=0):
    model.eval()
    total_loss = 0.0
    agg_metrics = defaultdict(float)
    all_preds = []
    all_targets = []
    conf_matrix = np.zeros((num_classes - 1, num_classes - 1), dtype=np.int64)
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            batch_cm = compute_confusion(preds, masks, num_classes, ignore_index)
            conf_matrix += batch_cm
            metrics = compute_metrics(preds.flatten(), masks.flatten(), num_classes, ignore_index)
            for k, v in metrics.items():
                if k != 'per_class':
                    agg_metrics[k] += v
            num_batches += 1

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = {k: agg_metrics[k] / num_batches for k in agg_metrics}

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    valid_idx = all_targets != ignore_index
    cls_report = classification_report(all_targets[valid_idx], all_preds[valid_idx],
                                       labels=[1, 2, 3, 4],
                                       target_names=["Barren", "LowVeg", "AutumnOlive", "OtherTree"],
                                       output_dict=True)
    return avg_loss, avg_metrics, conf_matrix, cls_report


# training loop

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir, stage, model_type, patience=15):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'confusion_matrix': [],
        'classification_report': []
    }

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    best_val_loss = float('inf')
    best_epoch = 0
    best_val_metrics = None
    no_improve_epochs = 0

    log_file = os.path.join(output_dir, f"training_log_{model_type}_{stage}.txt")
    with open(log_file, 'w') as f:
        f.write(f"starting training for {model_type} model on {stage} stage\n")
        f.write(f"epochs: {epochs}, patience: {patience}\n")
        f.write("epoch,train_loss,val_loss,val_accuracy,val_f1\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        nan_batches = 0
        processed_batches = 0

        for images, masks in train_loader:
            if torch.isnan(images).any():
                images = torch.nan_to_num(images)
                nan_batches += 1

            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            if torch.isnan(outputs).any():
                nan_batches += 1
                continue

            try:
                loss = criterion(outputs, masks)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    nan_batches += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item() * images.size(0)
                processed_batches += 1

            except Exception:
                nan_batches += 1
                continue

        if nan_batches > len(train_loader) // 2:
            with open(log_file, 'a') as f:
                f.write(f"too many nan batches ({nan_batches}) at epoch {epoch + 1}, stopping\n")
            break

        processed_samples = max(processed_batches * batch_size, 1)
        train_loss = running_loss / processed_samples
        history['train_loss'].append(train_loss)

        val_loss, val_metrics, conf_mat, cls_report = evaluate_model(model, val_loader, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics.get('accuracy', 0))
        history['val_precision'].append(val_metrics.get('precision', 0))
        history['val_recall'].append(val_metrics.get('recall', 0))
        history['val_f1'].append(val_metrics.get('f1', 0))
        history['confusion_matrix'].append(conf_mat.tolist())
        history['classification_report'].append(cls_report)

        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},{val_metrics.get('accuracy', 0):.6f},{val_metrics.get('f1', 0):.6f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics

            if isinstance(model, parallel.DataParallel):
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()

            model_path = os.path.join(output_dir, f"unet_{model_type}_{stage}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'metrics': best_val_metrics,
                'model_config': {
                    'in_channels': get_input_channels(model_type),
                    'num_filters': num_filters,
                    'kernel_size': kernel_size,
                    'num_classes': num_classes,
                    'dropout_rate': dropout_rate
                }
            }, model_path)

            with open(log_file, 'a') as f:
                f.write(f"saved best model at epoch {epoch + 1} with val_loss {val_loss:.6f}\n")

            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            with open(log_file, 'a') as f:
                f.write(f"early stopping triggered after {epoch + 1} epochs\n")
            break

    with open(log_file, 'a') as f:
        f.write(f"best model was from epoch {best_epoch + 1} with val_loss {best_val_loss:.6f}\n")

    return model, history, best_epoch


# helper - get_input_channels and init_weights imported from unet_model.py


# entry point

def run_training():

    if not output_dir:
        if model_type == 'rgb':
            folder = "rgb_chips"
        elif model_type == 'multispectral':
            folder = "multispec_chips"
        elif model_type == 'vi_composite':
            folder = "vi_comp_chips_corrected"
        output_dir_final = os.path.join(base_dir, folder, "results", stage)
    else:
        output_dir_final = output_dir

    os.makedirs(output_dir_final, exist_ok=True)

    dataset = UNetSegmentationDataset(
        base_dir=base_dir,
        stage=stage,
        model_type=model_type,
        transform=None,
        mask_transform=None,
        augment=augment,
        debug_subset=False
    )

    print(f"loaded dataset: {len(dataset)} samples")

    train_idx, val_idx = stratified_split(dataset)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")

    class_counts = calculate_class_distribution(dataset, train_idx)
    class_weights = compute_class_weights(class_counts)

    mask_weights = {}
    for idx in train_idx:
        try:
            with rasterio.open(dataset.mask_files[idx]) as src:
                mask = src.read(1).astype(np.int64)
                weight = 1.0
                for cls, priority in class_priorities.items():
                    if (mask == cls).any():
                        weight = max(weight, priority)
                mask_weights[idx] = weight
        except Exception:
            mask_weights[idx] = 1.0

    train_weights = [mask_weights.get(i, 1.0) for i in train_idx]
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    num_gpus = torch.cuda.device_count()
    num_workers_final = num_workers * max(num_gpus, 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers_final)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers_final)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    in_channels = get_input_channels(model_type)
    model = UNet(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        activation=nn.ReLU,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)

    model.apply(init_weights)

    # dataparallel for multi-gpu
    if num_gpus > 1:
        model = parallel.DataParallel(model)
        print(f"using {num_gpus} gpus")

    criterion = WeightedCrossEntropyFocalLoss(
        alpha=class_weights,
        gamma=focal_gamma,
        ignore_index=0,
        ce_weight=ce_weight,
        focal_weight=focal_weight,
        reduction='mean'
    )

    optimizer = optim.AdamW(model.parameters(),
                            lr=initial_lr,
                            weight_decay=weight_decay)

    print(f"\ntraining configuration:")
    print(f"  model type: {model_type} ({in_channels} channels)")
    print(f"  stage: {stage}")
    print(f"  batch size: {batch_size}")
    print(f"  architecture: {num_filters} filters, kernel {kernel_size}, dropout {dropout_rate}")
    print(f"  learning rate: {initial_lr} -> {max_lr}")
    print(f"  class counts: {class_counts}")
    print(f"  class weights: {[f'{w:.3f}' for w in class_weights]}")
    print(f"  output directory: {output_dir_final}")

    model, history, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs, output_dir_final, stage, model_type, patience
    )

    if model_type == 'rgb':
        folder_suffix = 'rgb'
    elif model_type == 'multispectral':
        folder_suffix = 'multispec'
    elif model_type == 'vi_composite':
        folder_suffix = 'vicomp'

    torch.save(history, os.path.join(output_dir_final, f"metrics_{folder_suffix}_{stage}.pth"))

    loss_report_file = os.path.join(output_dir_final, f"loss_{folder_suffix}_{stage}.txt")
    eval_report_file = os.path.join(output_dir_final, f"evaluation_{folder_suffix}_{stage}.txt")

    with open(loss_report_file, 'w') as f:
        f.write("train loss per epoch:\n" + "\n".join(map(str, history['train_loss'])) + "\n")
        f.write("validation loss per epoch:\n" + "\n".join(map(str, history['val_loss'])) + "\n")

    output_lines = []
    output_lines.append(f"model type: {model_type}")
    output_lines.append(f"stage: {stage}")
    output_lines.append(f"best model saved from epoch {best_epoch + 1}")
    output_lines.append(f"class counts: {class_counts}")
    output_lines.append(f"class weights: {class_weights}")
    output_lines.append(f"class priorities: {class_priorities}")
    output_lines.append("\nvalidation metrics per epoch:")

    for i in range(len(history['val_loss'])):
        output_lines.append(f"epoch {i + 1}: loss: {history['val_loss'][i]:.4f}, "
                            f"accuracy: {history['val_accuracy'][i]:.4f}, "
                            f"precision: {history['val_precision'][i]:.4f}, "
                            f"recall: {history['val_recall'][i]:.4f}, "
                            f"f1: {history['val_f1'][i]:.4f}")

    if best_epoch < len(history['confusion_matrix']):
        output_lines.append("\nbest epoch confusion matrix:")
        output_lines.append(str(history['confusion_matrix'][best_epoch]))
        output_lines.append("\nbest epoch classification report:")
        output_lines.append(str(history['classification_report'][best_epoch]))

    with open(eval_report_file, 'w') as f:
        f.write("\n".join(output_lines))

    print(f"\ntraining completed")
    print(f"best epoch: {best_epoch + 1}")
    print(f"results saved to: {output_dir_final}")

    return history


if __name__ == '__main__':
    run_training()
