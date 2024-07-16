import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import csv
import os
 

def load_and_combine_features_local(modality_keys, dataset_type):
    combined_features = []
    for modality in modality_keys:
        # Load the features for each modality
        features = np.load(f'../local_spatial/features/{modality}/{dataset_type}/{dataset_type}_backbone_outputs.npy')
        combined_features.append(features)
    # Combine features along the feature dimension (axis=1)
    return np.concatenate(combined_features, axis=1)

def make_csv(y_pred_validation, modality_used):
    df = pd.read_csv('dataset/MICCAI_BraTS2020_ValidationData/survival_evaluation.csv')
    validation_ids = df['BraTS20ID'].values
    filename = f"./global_predictions/Light_GBM/{modality_used}_Light_GBM.csv"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Days"])
        for id, day in zip(validation_ids, y_pred_validation):
            writer.writerow([id, day])

    print(f"CSV file '{filename}' created successfully.")

def load_features(modality_used):
    base_dir = os.path.join('extracted_features', modality_used)
    train_features = np.load(os.path.join(base_dir, 'train_features.npy'))
    validate_features = np.load(os.path.join(base_dir, 'validate_features.npy'))
    train_labels = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return train_features, validate_features, train_labels

def train_model(train_features, validate_features, train_labels, modality_used):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    validate_features_scaled = scaler.transform(validate_features)

    param_dist = {
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_samples': [10, 20, 30]
    }

    lgb_model = lgb.LGBMRegressor(random_state=42)
    random_search = RandomizedSearchCV(lgb_model, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(train_features_scaled, train_labels)

    y_pred_validation = random_search.predict(validate_features_scaled)
    make_csv(y_pred_validation, modality_used)

if __name__ == "__main__":
    modality_keys_list_global_feaatures = [
        ["flair"],
        ["t1ce"],
        ["flair", "t1ce"],
        ["flair", "t1ce", "t2"],
        ["flair", "t1", "t1ce", "t2"]
    ]
    modality_keys_list_local_feaatures = [
        ["flair"],
        ["t1ce"],
        ['t1'],
        ['t2'],
        ["flair", "t1ce"],
        ["flair", "t1ce", "t2"],
        ["flair", "t1", "t1ce", "t2"]
    ]
    for modality_keys in modality_keys_list_global_feaatures:
        modality_used = "_".join(modality_keys)
        train_features, validate_features, train_labels = load_features(modality_used)
        train_model(train_features, validate_features, train_labels, modality_used)
