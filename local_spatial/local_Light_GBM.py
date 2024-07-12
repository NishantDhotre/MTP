import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import os, re, csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define directories
train_path = '../dataset/MICCAI_BraTS2020_TrainingData/'
val_path = '../dataset/MICCAI_BraTS2020_ValidationData/'
modality_keys_list = [
    ["flair", "t1ce"],
    ["flair", "t1ce", "t2"],
    ["flair", "t1", "t1ce", "t2"]
]
BATCH_SIZE = 4
model_used = 'Light_GBM'

def make_csv(y_pred_validation, modality_used):
    df = pd.read_csv(os.path.join(val_path, 'survival_evaluation.csv'))
    validation_ids = df['BraTS20ID'].values
    filename = f"../local_predictions/{model_used}/{modality_used}_{model_used}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Days"])
        for id, day in zip(validation_ids, y_pred_validation):
            writer.writerow([id, day])
    print(f"CSV file '{filename}' created successfully.")

def preprocess_labels(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['Survival_days'] = df['Survival_days'].apply(lambda x: int(re.search(r'\d+', x).group()) if isinstance(x, str) else x)
    df['Survival_days'] = pd.to_numeric(df['Survival_days'], errors='coerce').dropna().astype(int)
    return df['Survival_days'].values, df['BraTS20ID'].values

def create_data_list_val(data_dir, modality_key):
    df = pd.read_csv(os.path.join(data_dir, 'survival_evaluation.csv'))
    patient_ids = df['BraTS20ID'].values
    data_list = []
    for patient in tqdm(patient_ids, desc="Creating validation data list"):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_path = f"{patient}_{modality_key}.nii"
            data_list.append(data_path)
    return data_list

def load_and_combine_features(modality_keys, dataset_type):
    combined_features = []
    for modality in modality_keys:
        features = np.load(f'./features/{modality}/{dataset_type}/{dataset_type}_backbone_outputs.npy')
        combined_features.append(features)
    return np.concatenate(combined_features, axis=1)

# Main loop to process each combination of modalities
for modality_keys in modality_keys_list:
    print(f"\nProcessing modality combination: {modality_keys}")
    
    # Load the combined features
    print("Loading and combining features...")
    train_features = load_and_combine_features(modality_keys, 'train')
    validation_features = load_and_combine_features(modality_keys, 'validation')

    # Load the corresponding labels
    print("Loading labels...")
    train_labels, train_id = preprocess_labels(os.path.join(train_path, 'survival_info.csv'))

    # Feature selection
    print("Performing feature selection...")
    selector = SelectKBest(f_regression, k=20 * len(modality_keys))  # Adjust k based on number of modalities
    train_features_selected = selector.fit_transform(train_features, train_labels)
    validation_features_selected = selector.transform(validation_features)

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_selected)
    validation_features_scaled = scaler.transform(validation_features_selected)

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'num_leaves': [7, 15, 31],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Initialize the LGBMRegressor
    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)

    # Perform RandomizedSearchCV
    print("Performing RandomizedSearchCV...")
    random_search = RandomizedSearchCV(lgb_model, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    random_search.fit(train_features_scaled, train_labels)

    # Print the best parameters and score
    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score (RMSE): {:.2f}".format(np.sqrt(-random_search.best_score_)))

    # Use the best model to make predictions
    best_model = random_search.best_estimator_

    # Predict and evaluate on a held-out set
    X_train, X_val, y_train, y_val = train_test_split(train_features_scaled, train_labels, test_size=0.2, random_state=42)
    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    print(f'RMSE on held-out set: {rmse}')

    # Predict on the validation set
    print("Predicting on validation set...")
    validation_pred = best_model.predict(validation_features_scaled)

    # Generate unique identifier for this combination
    combined_modality_key = '_'.join(modality_keys)

    make_csv(validation_pred, combined_modality_key)

    # Save the trained model
    os.makedirs('./GBM_model', exist_ok=True)
    best_model.booster_.save_model(f'./GBM_model/{combined_modality_key}_{model_used}_model.txt')
    print(f"Model saved to ./GBM_model/{combined_modality_key}_{model_used}_model.txt")

    # Print feature importances
    feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_, range(train_features_scaled.shape[1]))), columns=['Value','Feature'])
    print("Feature Importances:")
    print(feature_imp)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
    plt.title(f'Feature Importances ({combined_modality_key})')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    os.makedirs('./GBM_model/features_importance', exist_ok=True)
    plt.savefig(f'./GBM_model/features_importance/{combined_modality_key}_feature_importances.png')
    print(f"Feature importance plot saved as './GBM_model/features_importance/{combined_modality_key}_feature_importances.png'")

    print(f"Finished processing modality combination: {modality_keys}\n")