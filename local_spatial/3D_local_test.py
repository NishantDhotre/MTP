import torch
import os
import nibabel as nib
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import re
import pandas as pd

# Constants
MODALITY = 't1'
TRAIN_PATH = '../dataset/MICCAI_BraTS2020_TrainingData/'
VAL_PATH = '../dataset/MICCAI_BraTS2020_ValidationData/'
BATCH_SIZE = 4
FEATURE_SIZE = 64
 

 

class Conv4_3D(torch.nn.Module):
    """A simple 4 layers 3D CNN."""
    def __init__(self):
        super(Conv4_3D, self).__init__()
        self.feature_size = 64
        self.name = "conv4_3d"

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(),
            torch.nn.AvgPool3d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            torch.nn.AvgPool3d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool3d(kernel_size=2, stride=2)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.flatten(h)
        return h

# Data transformations
class ToTensor3D:
    def __call__(self, volume):
        return torch.from_numpy(volume).float().unsqueeze(0)
class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, volume):
        for t in self.transforms:
            volume = t(volume)
        return volume

class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, volume):
        return (volume - self.mean) / self.std

# Data handling functions
def create_data_list(data_dir, patient_ids,  modality_keys):
    data_list = []
    for idx, patient in enumerate(patient_ids):
        patient_dir = f"{data_dir}{patient}/"
        if os.path.isdir(patient_dir):
            data_dict = f"{data_dir}{patient}/{patient}_{modality_keys}.nii"
            data_list.append(data_dict)
    return data_list 

class MRIDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.volumes = []
        
        for file in tqdm(file_list, desc="Loading files"):
            img = nib.load(file).get_fdata()
            self.volumes.append(img)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        img = self.volumes[index]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float().unsqueeze(0)
        return img

# Feature extraction
def process_and_save_features(data_path, save_path, dataset_name, id):
    file_list = create_data_list(data_path, id, MODALITY)
    print("for ",dataset_name, len(file_list))
    dataset = MRIDataset(file_list=file_list, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    backbone_outputs = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Processing {dataset_name}"):
            features = backbone(data)
            backbone_outputs.append(features.cpu().numpy())

    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f'{dataset_name}_backbone_outputs.npy'), np.concatenate(backbone_outputs, axis=0))
    print(f"Features for {dataset_name} saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Initialize the backbone model
    backbone = Conv4_3D()
    backbone.load_state_dict(torch.load('./backbone_mri_3d.tar'))
    backbone.eval()

    # Define the transformations
    normalize = Normalize3D(mean=0.5, std=0.5)
    transform = Compose3D([
    ToTensor3D(),
    normalize
    ])
    df = pd.read_csv(f'{TRAIN_PATH}survival_info.csv')
    train_id = df['BraTS20ID'].values
    df = pd.read_csv(f'{VAL_PATH}survival_evaluation.csv')
    val_id = df['BraTS20ID'].values
    print("train id", len(train_id))
    print("val id", len(val_id))
    # Process and save features for test and validation sets
    process_and_save_features(TRAIN_PATH, f'./features/{MODALITY}/train', 'train', train_id)
    process_and_save_features(VAL_PATH, f'./features/{MODALITY}/validation', 'validation', val_id)
