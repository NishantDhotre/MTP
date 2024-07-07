import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import os, re, csv
from tqdm import tqdm
from scipy.stats import loguniform
import matplotlib.pyplot as plt

# Define directories
train_path = '../dataset/MICCAI_BraTS2020_TrainingData/'
val_path = '../dataset/MICCAI_BraTS2020_ValidationData/'
modality_key = 't2'
BATCH_SIZE = 4
model_used = 'XGB_Regression'

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

# Load the saved features
print("Loading features...")
train_features = np.load(f'./features/{modality_key}/train/train_backbone_outputs.npy')
validation_features = np.load(f'./features/{modality_key}/validation/validation_backbone_outputs.npy')

# Load the corresponding labels
print("Loading labels...")
train_labels, train_id = preprocess_labels(os.path.join(train_path, 'survival_info.csv'))

# Feature selection
print("Performing feature selection...")
selector = SelectKBest(f_regression, k=20)  # Select top 20 features
train_features_selected = selector.fit_transform(train_features, train_labels)
validation_features_selected = selector.transform(validation_features)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_selected)
validation_features_scaled = scaler.transform(validation_features_selected)

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': loguniform(1e-3, 1),  # Log-uniform distribution
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Initialize the XGBRegressor with early stopping
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=10)

# Perform RandomizedSearchCV with cross-validation
print("Performing RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=5, 
    random_state=42, 
    n_jobs=-1, 
    verbose=2, 
    scoring='neg_mean_squared_error'
)

# Split the data for early stopping
X_train, X_early_stop, y_train, y_early_stop = train_test_split(
    train_features_scaled, train_labels, test_size=0.2, random_state=42
)

random_search.fit(
    X_train, 
    y_train, 
    eval_set=[(X_early_stop, y_early_stop)],
    verbose=False
)

# Print the best parameters and score
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score (RMSE): {:.2f}".format(np.sqrt(-random_search.best_score_)))

# Use the best model to make predictions
best_model = random_search.best_estimator_

# Debug information
print(f"Features shape: {train_features_scaled.shape}")
print(f"Labels shape: {train_labels.shape}")
print(f"Features dtype: {train_features_scaled.dtype}")
print(f"Labels dtype: {train_labels.dtype}")
print(f"NaN in features: {np.isnan(train_features_scaled).any()}")
print(f"Inf in features: {np.isinf(train_features_scaled).any()}")
print(f"NaN in labels: {np.isnan(train_labels).any()}")
print(f"Inf in labels: {np.isinf(train_labels).any()}")

# Try fitting on a sample
X_sample, y_sample = train_features_scaled[:1000], train_labels[:1000]
try:
    best_model.fit(X_sample, y_sample)
    print("Sample fit successful")
except Exception as e:
    print(f"Error fitting the model on sample: {e}")

# Perform cross-validation for more robust evaluation
try:
    cv_scores = cross_val_score(best_model, train_features_scaled, train_labels, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    rmse_scores = np.sqrt(-cv_scores)
    print(f'Cross-validation RMSE scores: {rmse_scores}')
    print(f'Mean RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores) * 2:.2f})')
except Exception as e:
    print(f"Error in cross-validation: {e}")
    print("Trying with base model...")
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    cv_scores = cross_val_score(base_model, train_features_scaled, train_labels, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    rmse_scores = np.sqrt(-cv_scores)
    print(f'Cross-validation RMSE scores with base model: {rmse_scores}')
    print(f'Mean RMSE with base model: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores) * 2:.2f})')

# Predict on the validation set
print("Predicting on validation set...")
validation_pred = best_model.predict(validation_features_scaled)

make_csv(validation_pred, modality_key)

# Save the trained model
best_model.save_model(f'./XGB_model/{modality_key}_{model_used}_model.json')
print(f"Model saved to ./XGB_model/{modality_key}_{model_used}_model.json")

# Print feature importances
feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_, range(train_features_scaled.shape[1]))), columns=['Value', 'Feature'])
print("Feature Importances:")
print(feature_imp)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
os.makedirs('./XGB_model/features_importance/', exist_ok=True)
plt.savefig(f'./XGB_model/features_importance/{modality_key}_feature_importances.png')
print(f"Feature importance plot saved as './local_spatial/XGB_model/features_importance/{modality_key}_feature_importances.png'")