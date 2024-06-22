import os
import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped, DivisiblePadd, ResizeWithPadOrCropd
)
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import DivisiblePad
from monai.data.image_reader import NibabelReader
import pty
from sklearn.metrics import mean_squared_error

pty.fork = lambda: (0, 0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

# Set deterministic training for reproducibility
set_determinism(seed=0)

# modality_keys = ["flair", "t1", "t1ce", "t2"]
# 'dataset/MICCAI_BraTS2020_TrainingData'

# Define directories
train_path = 'dataset/MICCAI_BraTS2020_TrainingData/'
val_path = 'dataset/MICCAI_BraTS2020_ValidationData/'
# Function to create a list of data dictionaries
def create_data_list(data_dir, modality_keys):
    data_list = []
    patients = os.listdir(data_dir)
    for patient in patients:
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_dict = {key: os.path.join(patient_dir, f"{patient}_{key}.nii") for key in modality_keys}
            data_list.append(data_dict)
    return data_list
# Define transformations for training and validation
def get_transforms(modality_keys, pixdim=(1.0, 1.0, 1.0)):
    transforms = Compose(
        [
            LoadImaged(keys=modality_keys, reader=NibabelReader()),
            EnsureChannelFirstd(keys=modality_keys),
            Spacingd(keys=modality_keys, pixdim=pixdim, mode=("bilinear")),
            Orientationd(keys=modality_keys, axcodes="RAS"),
            ScaleIntensityRanged(keys=modality_keys, a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=modality_keys, source_key=modality_keys[0], allow_smaller=True),
            ResizeWithPadOrCropd(keys=modality_keys, spatial_size=(256, 256, 160)),  # Pad to nearest multiple of 32
            RandFlipd(keys=modality_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=modality_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=modality_keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=modality_keys, prob=0.5, max_k=3),
            RandShiftIntensityd(keys=modality_keys, offsets=0.10, prob=0.5),
            EnsureTyped(keys=modality_keys),
        ]
    )
    return transforms
def get_val_transforms(modality_keys, pixdim=(1.0, 1.0, 1.0)):
    transforms = Compose(
        [
            LoadImaged(keys=modality_keys, reader=NibabelReader()),
            EnsureChannelFirstd(keys=modality_keys),
            Spacingd(keys=modality_keys, pixdim=pixdim, mode=("bilinear")),
            Orientationd(keys=modality_keys, axcodes="RAS"),
            ScaleIntensityRanged(keys=modality_keys, a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=modality_keys, source_key=modality_keys[0], allow_smaller=True),
            ResizeWithPadOrCropd(keys=modality_keys, spatial_size=(256, 256, 160)),  # Pad to nearest multiple of 32
            EnsureTyped(keys=modality_keys),
        ]
    )
    return transforms

# Training function
def train_model(modality_keys, train_path, val_path, max_epochs=10, val_interval=2):
    in_channels = len(modality_keys)
    out_channels = len(modality_keys)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SwinUNETR(
        img_size=(256, 256, 160),  # Adjusted image size
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)
    
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    train_transforms = get_transforms(modality_keys)
    train_data_list = create_data_list(train_path, modality_keys)

    val_transforms = get_val_transforms(modality_keys)
    val_data_list = create_data_list(val_path, modality_keys)

    # Create datasets and dataloaders
    train_ds = CacheDataset(
        data=train_data_list,
        transform=train_transforms,
        cache_rate=0.5,
        num_workers=4,
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(
        data=val_data_list,
        transform=val_transforms,
        cache_rate=0.5,
        num_workers=4,
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
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
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = torch.cat([val_data[key] for key in modality_keys], dim=1).to(device)
                    val_outputs = model(val_inputs)
                    val_loss = loss_function(val_outputs, val_inputs)
                    print(f"Validation loss at epoch {epoch + 1}: {val_loss.item():.4f}")

        modality_used = "_".join(modality_keys)
        model_save_path = f"model_saved/swin_unetr_{modality_used}_epoch.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Save the model state dictionary to a file
    modality_used = "_".join(modality_keys)
    model_save_path = f"model_saved/swin_unetr_{modality_used}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
# modality_keys = ["flair", "t1", "t1ce", "t2"]
modality_keys = ["flair"]
train_model(modality_keys=modality_keys, train_path=train_path, val_path=val_path, max_epochs=10, val_interval=2)
# modality_keys = ["flair", "t1", "t1ce", "t2"]
modality_keys = ["t1ce"]
train_model(modality_keys=modality_keys, train_path=train_path, val_path=val_path, max_epochs=10, val_interval=2)

# modality_keys = ["flair", "t1", "t1ce", "t2"]
modality_keys = ["flair", "t1ce"]
train_model(modality_keys=modality_keys, train_path=train_path, val_path=val_path, max_epochs=10, val_interval=2)

# modality_keys = ["flair", "t1", "t1ce", "t2"]
modality_keys = ["flair", "t1ce", "t2"]
train_model(modality_keys=modality_keys, train_path=train_path, val_path=val_path, max_epochs=10, val_interval=2)

# modality_keys = ["flair", "t1", "t1ce", "t2"]
modality_keys = ["flair", "t1", "t1ce", "t2"]
train_model(modality_keys=modality_keys, train_path=train_path, val_path=val_path, max_epochs=10, val_interval=2)
