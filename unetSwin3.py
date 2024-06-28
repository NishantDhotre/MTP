import os
import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped, ResizeWithPadOrCropd
)
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism
from monai.data.image_reader import NibabelReader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Constants
TRAIN_PATH = 'dataset/MICCAI_BraTS2020_TrainingData/'
VAL_PATH = 'dataset/MICCAI_BraTS2020_ValidationData/'
MAX_EPOCHS = 10
VAL_INTERVAL = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1  # Kept at 1 for older GPUs
NUM_WORKERS = 2
PATIENCE = 5

def set_seed(seed=0):
    set_determinism(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def create_data_list(data_dir, modality_keys):
    data_list = []
    patients = os.listdir(data_dir)
    for patient in patients:
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_dict = {key: os.path.join(patient_dir, f"{patient}_{key}.nii") for key in modality_keys}
            data_list.append(data_dict)
    return data_list

def get_transforms(modality_keys, pixdim=(1.0, 1.0, 1.0), is_train=True):
    transform_list = [
        LoadImaged(keys=modality_keys, reader=NibabelReader()),
        EnsureChannelFirstd(keys=modality_keys),
        Spacingd(keys=modality_keys, pixdim=pixdim, mode=("bilinear")),
        Orientationd(keys=modality_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=modality_keys, a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=modality_keys, source_key=modality_keys[0], allow_smaller=True),
        ResizeWithPadOrCropd(keys=modality_keys, spatial_size=(256, 256, 160)),  # Kept original size
        EnsureTyped(keys=modality_keys),
    ]
    
    if is_train:
        transform_list.extend([
            RandFlipd(keys=modality_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=modality_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=modality_keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=modality_keys, prob=0.5, max_k=3),
            RandShiftIntensityd(keys=modality_keys, offsets=0.10, prob=0.5),
        ])
    
    return Compose(transform_list)

def train_model(modality_keys, train_path, val_path, max_epochs=MAX_EPOCHS, val_interval=VAL_INTERVAL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SwinUNETR(
        img_size=(256, 256, 160),  # Kept original size
        in_channels=len(modality_keys),
        out_channels=len(modality_keys),
        feature_size=24,  # Reduced from 48 to save memory
        use_checkpoint=True,
    )
    
    model.to(device)
    
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    train_transforms = get_transforms(modality_keys, is_train=True)
    train_data_list = create_data_list(train_path, modality_keys)

    val_transforms = get_transforms(modality_keys, is_train=False)
    val_data_list = create_data_list(val_path, modality_keys)

    train_ds = CacheDataset(data=train_data_list, transform=train_transforms, cache_rate=0.1, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_rate=0.1, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    best_metric = float('inf')
    best_metric_epoch = -1
    epoch_loss_values = []
    patience_counter = 0
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = torch.cat([batch_data[key] for key in modality_keys], dim=1).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch == 1 :
            # best_metric = val_loss
            # best_metric_epoch = epoch + 1
            modality_used = "_".join(modality_keys)
            model_save_path = f"model_saved/swin_unetr_{modality_used}_best.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with validation loss: {best_metric:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation"):
                    val_inputs = torch.cat([val_data[key] for key in modality_keys], dim=1).to(device)
                    val_outputs = model(val_inputs)
                    val_loss += loss_function(val_outputs, val_inputs).item()
            val_loss /= len(val_loader)
            print(f"Validation loss at epoch {epoch + 1}: {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_metric:
                best_metric = val_loss
                best_metric_epoch = epoch + 1
                modality_used = "_".join(modality_keys)
                model_save_path = f"model_saved/swin_unetr_{modality_used}_best.pth"
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path} with validation loss: {best_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                    break

    print(f"Training completed, best validation loss: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    modality_keys_list = [
        # ["flair"],
        # ["t1ce"],
        ["flair", "t1ce"],
        ["flair", "t1ce", "t2"],
        ["flair", "t1", "t1ce", "t2"]
    ]

    for modality_keys in modality_keys_list:
        print(f"Now training on modalities: {modality_keys}")
        train_model(modality_keys=modality_keys, train_path=TRAIN_PATH, val_path=VAL_PATH)