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
import pandas as pd
import numpy as np
import re
import torch.nn.functional as F
from tqdm import tqdm

# Define directories
train_path = 'dataset/MICCAI_BraTS2020_TrainingData/'
val_path = 'dataset/MICCAI_BraTS2020_ValidationData/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_data_list(data_dir, patient_ids, labels, modality_keys):
    data_list = []
    for idx, patient in enumerate(patient_ids):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_dict = {key: os.path.join(patient_dir, f"{patient}_{key}.nii") for key in modality_keys}
            data_dict['label'] = labels[idx]
            data_list.append(data_dict)
    return data_list

def create_data_list_val(data_dir, modality_keys):
    df = pd.read_csv(os.path.join(data_dir, 'survival_evaluation.csv'))
    patient_ids = df['BraTS20ID'].values
    data_list = []
    for patient in tqdm(patient_ids, desc="Creating validation data list"):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_dict = {key: os.path.join(patient_dir, f"{patient}_{key}.nii") for key in modality_keys}
            data_list.append(data_dict)
    return data_list

def preprocess_labels(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['Survival_days'] = df['Survival_days'].apply(lambda x: int(re.search(r'\d+', x).group()) if isinstance(x, str) else x)
    df['Survival_days'] = pd.to_numeric(df['Survival_days'], errors='coerce').dropna().astype(int)
    return df['Survival_days'].values, df['BraTS20ID'].values

def get_transforms(modality_keys, pixdim=(1.0, 1.0, 1.0), is_train=True):
    transform_list = [
        LoadImaged(keys=modality_keys, reader=NibabelReader()),
        EnsureChannelFirstd(keys=modality_keys),
        Spacingd(keys=modality_keys, pixdim=pixdim, mode=("bilinear")),
        Orientationd(keys=modality_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=modality_keys, a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=modality_keys, source_key=modality_keys[0], allow_smaller=True),
        ResizeWithPadOrCropd(keys=modality_keys, spatial_size=(256, 256, 160)),
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

class FeatureExtractorSwinUNETR(SwinUNETR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extract_features(self, x):
        hidden_states = self.swinViT(x, self.normalize)
        pooled_states = [F.adaptive_avg_pool3d(state, (4, 4, 4)) for state in hidden_states]
        return torch.cat(pooled_states, dim=1)

def extract_features_from_dataset(modality_keys, dataloader, feature_extractor):
    features = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        inputs = torch.cat([batch[key] for key in modality_keys], dim=1).to(device)
        with torch.no_grad():
            feature = feature_extractor.extract_features(inputs)
        feature = torch.mean(feature, dim=[2, 3, 4])
        features.append(feature.cpu().numpy())
    return np.concatenate(features)

def save_features(train_features, validate_features, train_labels, modality_used):
    base_dir = os.path.join('Global_extracted_features', modality_used)
    os.makedirs(base_dir, exist_ok=True)
    np.save(os.path.join(base_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(base_dir, 'validate_features.npy'), validate_features)
    np.save(os.path.join(base_dir, 'train_labels.npy'), train_labels)

if __name__ == "__main__":
    modality_keys_list = [
        ["flair"],
        ["t1ce"],
        ["flair", "t1ce"],
        ["flair", "t1ce", "t2"],
        ["flair", "t1", "t1ce", "t2"]
    ]
    train_labels, train_id = preprocess_labels(os.path.join(train_path, 'survival_info.csv'))
    for modality_keys in modality_keys_list:
        print("Now working on", modality_keys)
        train_data_list = create_data_list(train_path, train_id, train_labels, modality_keys)
        valdate_data_list = create_data_list_val(val_path, modality_keys)
        
        train_transforms = get_transforms(modality_keys, is_train=True)
        val_transforms = get_transforms(modality_keys, is_train=False)
        
        train_ds = CacheDataset(data=train_data_list, transform=train_transforms, cache_rate=0.5, num_workers=4)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

        validate_ds = CacheDataset(data=valdate_data_list, transform=val_transforms, cache_rate=0.5, num_workers=4)
        validate_loader = DataLoader(validate_ds, batch_size=2, shuffle=False, num_workers=2)

        model = SwinUNETR(img_size=(256, 256, 160), in_channels=len(modality_keys), out_channels=len(modality_keys), feature_size=24, use_checkpoint=True).to(device)
        model_save_path = f"model_saved/swin_unetr_{'_'.join(modality_keys)}_best.pth"
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        feature_extractor = FeatureExtractorSwinUNETR(img_size=(256, 256, 160), in_channels=len(modality_keys), out_channels=len(modality_keys), feature_size=24, use_checkpoint=True).to(device)
        feature_extractor.load_state_dict(model.state_dict())
        feature_extractor.eval()

        train_features = extract_features_from_dataset(modality_keys, train_loader, feature_extractor)
        validate_features = extract_features_from_dataset(modality_keys, validate_loader, feature_extractor)

        save_features(train_features, validate_features, train_labels, '_'.join(modality_keys))
