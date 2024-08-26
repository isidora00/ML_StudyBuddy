from sklearn.ensemble import RandomForestRegressor
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

# Define and fit Random Forest Regressor model with PCA
pipeline_rf = Pipeline([
    ('pca', PCA()),  
    ('regressor', RandomForestRegressor())
])

param_grid_rf = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'regressor__n_estimators': [50, 100, 200],  # Number of trees in the forest
    'regressor__max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'regressor__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'regressor__min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
}

grid_search_rf = model_selection.GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_model_rf = grid_search_rf.best_estimator_
y_dev_pred_rf = best_model_rf.predict(X_dev)
dev_mse_rf = metrics.mean_squared_error(y_dev, y_dev_pred_rf)
print("Development set Mean Squared Error (Random Forest): ", dev_mse_rf)

# Retrain Random Forest model on the combined train and dev sets
X_combined_rf = np.concatenate([X_train, X_dev], axis=0)
y_combined_rf = np.concatenate([y_train, y_dev], axis=0)
best_model_rf.fit(X_combined_rf, y_combined_rf)

# Test Random Forest model
y_test_pred_rf = best_model_rf.predict(X_test)
test_mse_rf = metrics.mean_squared_error(y_test, y_test_pred_rf)
print("Test set Mean Squared Error (Random Forest): ", test_mse_rf)

# Print best parameters and best score from GridSearchCV
print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
print("Best cross-validation score (negative MSE) for Random Forest: ", grid_search_rf.best_score_)

# Define a function to calculate accuracy with tolerance
def calculate_accuracy_with_tolerance(true_values, predictions, tolerance):
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    return np.mean(np.abs(true_values - predictions) <= tolerance)

# Define tolerance values to test
tolerances = np.linspace(0.1, 5.0, 50)
accuracies_rf = [calculate_accuracy_with_tolerance(y_test, y_test_pred_rf, tol) for tol in tolerances]

# Plot REC Curve for Random Forest model
plt.figure(figsize=(12, 8))
plt.plot(tolerances, accuracies_rf, color='red', lw=2, marker='o', label='Random Forest')
plt.xlabel('Tolerance')
plt.ylabel('Accuracy')
plt.title('Regression Error Characteristic (REC) Curve')
plt.grid(True)
plt.legend()
plt.show()

# Calculate REC AUC for Random Forest model
rec_auc_rf = np.trapz(accuracies_rf, tolerances)

print(f"REC AUC for Random Forest: {rec_auc_rf:.4f}")
