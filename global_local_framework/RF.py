import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import csv
from sklearn.linear_model import LassoCV
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os

model_used = 'RF'

RF_global = [
    "flair_t1ce_t2",
    "flair",
    "flair_t1ce_t2",
    "flair_t1ce_t2",
    "flair",
    "flair",
    "flair",
    "flair",
    "flair_t1ce"
]
RF_local = [
    "flair_t1ce",
    "flair",
    "flair",
    "flair_t1ce_t2",
    "t1",
    "flair_t1ce",
    "flair_t1ce_t2",
    "flair_t1_t1ce_t2",
    "flair_t1_t1ce_t2"
]



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
    filename = f"../local_global_predictions/lasso_feture_selection/{model_used}/{modality_used}_{model_used}.csv"

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


def save_model_and_parameters(model, modality_used, lasso_mask, params):
    model_dir = f"./models/lasso_feture_selection/{model_used}/{modality_used}/"
    ensure_directory_exists(model_dir)
    model_file = os.path.join(model_dir, 'model.joblib')
    params_file = os.path.join(model_dir, 'params.txt')
    lasso_file = os.path.join(model_dir, 'lasso_mask.npy')
    
    # Save the model
    joblib.dump(model, model_file)
    
    # Save the parameters
    with open(params_file, 'w') as file:
        file.write(f"Best parameters: {params}\n")
    
    # Save the selected features mask
    np.save(lasso_file, lasso_mask)
    
    print(f"Model, parameters, and Lasso mask saved successfully for modality {modality_used}.")
    
    
def train_model(train_features, validate_features, train_labels, modality_used):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    validate_features_scaled = scaler.transform(validate_features)
    # Print feature size before Lasso feature selection
    print(f"Size of features before Lasso: {train_features_scaled.shape}")

    # Lasso Feature Selection with increased regularization
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, -0.5, 30)).fit(train_features_scaled, train_labels)
    
    # Select non-zero coefficients
    mask = lasso.coef_ != 0
    train_features_selected = train_features_scaled[:, mask]
    validate_features_selected = validate_features_scaled[:, mask]

    # Check if any features were selected
    if train_features_selected.shape[1] == 0:
        print(f"No features selected for modality {modality_used}. Skipping this combination.")
        return

    # Print feature size after Lasso feature selection
    print(f"Size of features after Lasso: {train_features_selected.shape}")

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf_model = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(train_features_scaled, train_labels)

    y_pred_validation = random_search.predict(validate_features_selected)
    make_csv(y_pred_validation, modality_used)
    
    # Save the model, parameters, and Lasso mask
    save_model_and_parameters(random_search.best_estimator_, modality_used, mask, random_search.best_params_)

 

for modality_used_global, modality_used_local in zip(RF_global, RF_local):
    modality_key_local = modality_used_local.split("_")
    modality_keys_list_global_features = modality_used_global.split("_")
    print(f"\nLoading and combining features... \n local-{modality_used_local}\n global-{modality_used_global}")
    local_train_features = load_and_combine_features(modality_key_local, 'train')
    local_validation_features = load_and_combine_features(modality_key_local, 'validation')

    global_train_features, global_validate_features, train_labels = load_features(modality_used_global)

    combined_training_features = np.concatenate((global_train_features, local_train_features), axis=1)
    combined_validation_features = np.concatenate((global_validate_features, local_validation_features), axis=1)

    modality_used = 'global_' + modality_used_global + '___local_' + modality_used_local
    train_model(combined_training_features, combined_validation_features, train_labels, modality_used)
