import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import csv
import os

model_used = 'XGB'

def ensure_directory_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def load_and_combine_features(modality_keys, dataset_type):
    combined_features = []
    for modality in modality_keys:
        # Load the features for each modality
        features = np.load(f'../local_spatial/features/{modality}/{dataset_type}/{dataset_type}_backbone_outputs.npy')
        combined_features.append(features)
    # Combine features along the feature dimension (axis=1)
    return np.concatenate(combined_features, axis=1)

def make_csv(y_pred_validation, modality_used):
    df = pd.read_csv('../dataset/MICCAI_BraTS2020_ValidationData/survival_evaluation.csv')
    validation_ids = df['BraTS20ID'].values
    filename = f"../local_global_predictions/{model_used}/{modality_used}_{model_used}.csv"

    ensure_directory_exists(filename)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Days"])
        for id, day in zip(validation_ids, y_pred_validation):
            writer.writerow([id, day])

    print(f"CSV file '{filename}' created successfully.")

def load_features(modality_used):
    base_dir = os.path.join('../Global_extracted_features', modality_used)
    train_features = np.load(os.path.join(base_dir, 'train_features.npy'))
    validate_features = np.load(os.path.join(base_dir, 'validate_features.npy'))
    train_labels = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return train_features, validate_features, train_labels

def train_model(train_features, validate_features, train_labels, modality_used):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    validate_features_scaled = scaler.transform(validate_features)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    xgb_model = xgb.XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(train_features_scaled, train_labels)

    y_pred_validation = random_search.predict(validate_features_scaled)
    make_csv(y_pred_validation, modality_used)

# Define modality keys
modality_keys_list_global_features = [
    ["flair"],
    ["t1ce"],
    ["flair", "t1ce"],
    ["flair", "t1ce", "t2"],
    ["flair", "t1", "t1ce", "t2"]
]

modality_keys_list_local_features = [
    ["flair"],
    ["t1ce"],
    ['t1'],
    ['t2'],
    ["flair", "t1ce"],
    ["flair", "t1ce", "t2"],
    ["flair", "t1", "t1ce", "t2"]
]

for modality_key_global in modality_keys_list_global_features:
    for modality_key_local in modality_keys_list_local_features:
        # Load the combined features
        modality_used_local = "_".join(modality_key_local)
        print("Loading and combining features...")
        local_train_features = load_and_combine_features(modality_key_local, 'train')
        local_validation_features = load_and_combine_features(modality_key_local, 'validation')

        modality_used_global = "_".join(modality_key_global)
        global_train_features, global_validate_features, train_labels = load_features(modality_used_global)

        combined_training_features = np.concatenate((global_train_features, local_train_features), axis=1)
        combined_validation_features = np.concatenate((global_validate_features, local_validation_features), axis=1)

        modality_used = 'global_' + modality_used_global + '___local_' + modality_used_local
        train_model(combined_training_features, combined_validation_features, train_labels, modality_used)
