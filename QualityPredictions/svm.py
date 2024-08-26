from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn import linear_model
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

y_train = np.array(y_train).ravel()
y_dev = np.array(y_dev).ravel()
y_test = np.array(y_test).ravel()

# Standardize features
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Define and fit SVM model with PCA
pipeline_svm = Pipeline([
    ('pca', PCA()),  
    ('svr', SVR())
])

param_grid_svm = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.2, 0.5],
    'svr__gamma' : ['scale','auto']
}

grid_search_svm = model_selection.GridSearchCV(pipeline_svm, param_grid_svm, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

best_model_svm = grid_search_svm.best_estimator_
y_dev_pred_svm = best_model_svm.predict(X_dev)
dev_mse_svm = metrics.mean_squared_error(y_dev, y_dev_pred_svm)
print("Development set Mean Squared Error (SVM): ", dev_mse_svm)
print("Best parameters found: ", grid_search_svm.best_params_)

# Retrain SVM model on the combined train and dev sets
X_combined_svm = np.concatenate([X_train, X_dev], axis=0)
y_combined_svm = np.concatenate([y_train, y_dev], axis=0)
best_model_svm.fit(X_combined_svm, y_combined_svm)

# Test SVM model
y_test_pred_svm = best_model_svm.predict(X_test)
test_mse_svm = metrics.mean_squared_error(y_test, y_test_pred_svm)
print("Test set Mean Squared Error (SVM): ", test_mse_svm)

# Print best parameters and best score from GridSearchCV for SVM
print("Best parameters for SVM: ", grid_search_svm.best_params_)
print("Best cross-validation score (negative MSE) for SVM: ", grid_search_svm.best_score_)

# Define a function to calculate accuracy with tolerance
def calculate_accuracy_with_tolerance(true_values, predictions, tolerance):
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    return np.mean(np.abs(true_values - predictions) <= tolerance)

# Define tolerance values to test
tolerances = np.linspace(0.1, 5.0, 50)
accuracies_svm = [calculate_accuracy_with_tolerance(y_test, y_test_pred_svm, tol) for tol in tolerances]

# Plot REC Curve for all models
plt.figure(figsize=(12, 8))
plt.plot(tolerances, accuracies_svm, color='red', lw=2, marker='o', label='SVM')
plt.xlabel('Tolerance')
plt.ylabel('Accuracy')
plt.title('Regression Error Characteristic (REC) Curve')
plt.grid(True)
plt.legend()
plt.show()

# Calculate REC AUC for all models
rec_auc_svm = np.trapz(accuracies_svm, tolerances)

print(f"REC AUC for SVM: {rec_auc_svm:.4f}")
