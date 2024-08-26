from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)
df = wine_quality.data

# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(df.features, df.targets, test_size=0.25, random_state=42, stratify=df.targets)
X_train, X_dev, y_train, y_dev = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Convert target variables to numpy arrays and flatten them
y_train = np.array(y_train).ravel()
y_dev = np.array(y_dev).ravel()
y_test = np.array(y_test).ravel()

# Standardize features
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Define and fit AdaBoost Regressor model with PCA
pipeline_ab = Pipeline([
    ('pca', PCA()),  
    ('regressor', AdaBoostRegressor())
])

param_grid_ab = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'regressor__n_estimators': [50, 100, 200],  # Number of boosting stages
    'regressor__learning_rate': [0.01, 0.1, 1.0],  # Learning rate shrinks the contribution of each tree
}

grid_search_ab = model_selection.GridSearchCV(pipeline_ab, param_grid_ab, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_ab.fit(X_train, y_train)

best_model_ab = grid_search_ab.best_estimator_
y_dev_pred_ab = best_model_ab.predict(X_dev)
dev_mse_ab = metrics.mean_squared_error(y_dev, y_dev_pred_ab)
print("Development set Mean Squared Error (AdaBoost): ", dev_mse_ab)

# Retrain AdaBoost model on the combined train and dev sets
X_combined_ab = np.concatenate([X_train, X_dev], axis=0)
y_combined_ab = np.concatenate([y_train, y_dev], axis=0)
best_model_ab.fit(X_combined_ab, y_combined_ab)

# Test AdaBoost model
y_test_pred_ab = best_model_ab.predict(X_test)
test_mse_ab = metrics.mean_squared_error(y_test, y_test_pred_ab)
print("Test set Mean Squared Error (AdaBoost): ", test_mse_ab)

# Print best parameters and best score from GridSearchCV
print("Best parameters for AdaBoost: ", grid_search_ab.best_params_)
print("Best cross-validation score (negative MSE) for AdaBoost: ", grid_search_ab.best_score_)

# Define a function to calculate accuracy with tolerance
def calculate_accuracy_with_tolerance(true_values, predictions, tolerance):
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    return np.mean(np.abs(true_values - predictions) <= tolerance)

# Define tolerance values to test
tolerances = np.linspace(0.1, 5.0, 50)
accuracies_ab = [calculate_accuracy_with_tolerance(y_test, y_test_pred_ab, tol) for tol in tolerances]

# Plot REC Curve for AdaBoost model
plt.figure(figsize=(12, 8))
plt.plot(tolerances, accuracies_ab, color='purple', lw=2, marker='o', label='AdaBoost')
plt.xlabel('Tolerance')
plt.ylabel('Accuracy')
plt.title('Regression Error Characteristic (REC) Curve')
plt.grid(True)
plt.legend()
plt.show()

# Calculate REC AUC for AdaBoost model
rec_auc_ab = np.trapz(accuracies_ab, tolerances)

print(f"REC AUC for AdaBoost: {rec_auc_ab:.4f}")
