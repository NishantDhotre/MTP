import torch
import numpy as np
import random
from scipy.ndimage import rotate, zoom
from torch.utils.data import Dataset
import nibabel as nib
import os
from tqdm import tqdm
import math

class RandomRotation3D:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, volume):
        angle = random.uniform(-self.degrees, self.degrees)
        axes = random.choice([(0, 1), (1, 2), (0, 2)])  # Randomly choose a plane to rotate
        return rotate(volume, angle, axes=axes, reshape=False)

class RandomResizedCrop3D:
    def __init__(self, output_size, scale=(0.8, 1.0)):
        self.output_size = output_size
        self.scale = scale

    def __call__(self, volume):
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        zoomed_volume = zoom(volume, scale_factor)
        crop_size = [min(self.output_size[i], zoomed_volume.shape[i]) for i in range(3)]
        start = [random.randint(0, zoomed_volume.shape[i] - crop_size[i]) for i in range(3)]
        cropped_volume = zoomed_volume[start[0]:start[0] + crop_size[0],
                                       start[1]:start[1] + crop_size[1],
                                       start[2]:start[2] + crop_size[2]]
        return zoom(cropped_volume, [self.output_size[i] / cropped_volume.shape[i] for i in range(3)])

class RandomHorizontalFlip3D:
    def __call__(self, volume):
        if random.random() > 0.5:
            return np.flip(volume, axis=random.choice([0, 1, 2])).copy()
        return volume

class ToTensor3D:
    def __call__(self, volume):
        return torch.from_numpy(volume).float().unsqueeze(0)

class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, volume):
        return (volume - self.mean) / self.std

class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, volume):
        for t in self.transforms:
            volume = t(volume)
        return volume

# Define the 3D transformations
normalize = Normalize3D(mean=0.5, std=0.5)
train_transform = Compose3D([
    RandomRotation3D(degrees=10),
    RandomResizedCrop3D(output_size=(128, 128, 128), scale=(0.8, 1.0)),
    RandomHorizontalFlip3D(),
    ToTensor3D(),
    normalize
])

class MRIDataset(Dataset):
    """Custom Dataset for loading MRI data as 3D volumes."""
    def __init__(self, file_list, K, transform=None):
        self.file_list = file_list
        self.K = K
        self.transform = transform
        self.volumes = []
        
        for file in tqdm(file_list, desc="Loading files"):
            img = nib.load(file).get_fdata()
            self.volumes.append(img)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        img = self.volumes[index]
        
        img_list = []
        if self.transform:
            for _ in range(self.K):
                img_transformed = self.transform(img)
                img_list.append(img_transformed)
        else:
            img_list = [torch.from_numpy(img).float().unsqueeze(0) for _ in range(self.K)]
        
        return img_list, 0  # 0 is a dummy target

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

class RelationalReasoning(torch.nn.Module):
    """Self-Supervised Relational Reasoning for 3D MRI data."""
    def __init__(self, backbone, feature_size=64):
        super(RelationalReasoning, self).__init__()
        self.backbone = backbone
        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size * 2, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1)
        )

    def aggregate(self, features, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        shifts_counter = 1
        for index_1 in range(0, size * K, size):
            for index_2 in range(index_1 + size, size * K, size):
                # Using the 'cat' aggregation function by default
                pos_pair = torch.cat([features[index_1:index_1 + size], features[index_2:index_2 + size]], 1)
                # Shuffle without collisions by rolling the mini-batch (negatives)
                neg_pair = torch.cat([features[index_1:index_1 + size], torch.roll(features[index_2:index_2 + size], shifts=shifts_counter, dims=0)], 1)
                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair)
                targets_list.append(torch.ones(size, dtype=torch.float32))
                targets_list.append(torch.zeros(size, dtype=torch.float32))
                shifts_counter += 1
                if shifts_counter >= size:
                    shifts_counter = 1  # avoid identity pairs
        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)
        return relation_pairs, targets

    def train_model(self, tot_epochs, train_loader):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()}
        ])
        BCE = torch.nn.BCEWithLogitsLoss()
        self.backbone.train()
        self.relation_head.train()
        for epoch in range(tot_epochs):
            for i, (data_augmented, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{tot_epochs}"):
                K = len(data_augmented)  # total augmentations
                x = torch.cat(data_augmented, 0)
                optimizer.zero_grad()
                # forward pass (backbone)
                features = self.backbone(x)
                # aggregation function
                relation_pairs, targets = self.aggregate(features, K)
                # forward pass (relation head)
                score = self.relation_head(relation_pairs).squeeze()
                # cross-entropy loss and backward
                loss = BCE(score, targets)
                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))

                if i % 100 == 0:
                    print(f'Batch [{i + 1}/{len(train_loader)}] - Loss: {loss.item():.5f}; Accuracy: {accuracy.item():.2f}%')

def create_data_list(data_dir):
    data_list = []
    patients = os.listdir(data_dir)
    for patient in tqdm(patients, desc="Creating data list"):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_dict = os.path.join(patient_dir, f"{patient}_flair.nii")
            data_list.append(data_dict)
    return data_list

# Paths and Hyper-parameters
train_path = '../dataset/MICCAI_BraTS2020_TrainingData/'
K = 4
batch_size = 4  # Reduced batch size due to 3D data's high memory consumption
tot_epochs = 10
feature_size = 64

# Initialize the backbone and the relational reasoning model
backbone = Conv4_3D()
model = RelationalReasoning(backbone, feature_size)

# Create the list of files and the dataset
file_list = create_data_list(train_path)
train_set = MRIDataset(file_list=file_list, K=K, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Train the model
model.train_model(tot_epochs=tot_epochs, train_loader=train_loader)
torch.save(model.backbone.state_dict(), './backbone_mri_3d.tar')
