from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
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

# Convert target variables to numpy arrays and flatten them
y_train = np.array(y_train).ravel()
y_dev = np.array(y_dev).ravel()
y_test = np.array(y_test).ravel()

# Standardize features
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# Define and fit Naive Bayes model with PCA
pipeline_nb = Pipeline([
    ('pca', PCA()),  
    ('classifier', GaussianNB())
])

param_grid_nb = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

grid_search_nb = model_selection.GridSearchCV(pipeline_nb, param_grid_nb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_nb.fit(X_train, y_train)

best_model_nb = grid_search_nb.best_estimator_
y_dev_pred_nb = best_model_nb.predict(X_dev)
dev_accuracy_nb = metrics.accuracy_score(y_dev, y_dev_pred_nb)
print("Development set Accuracy (Naive Bayes): ", dev_accuracy_nb)

# Retrain Naive Bayes model on the combined train and dev sets
X_combined_nb = np.concatenate([X_train, X_dev], axis=0)
y_combined_nb = np.concatenate([y_train, y_dev], axis=0)
best_model_nb.fit(X_combined_nb, y_combined_nb)

# Test Naive Bayes model
y_test_pred_nb = best_model_nb.predict(X_test)
test_accuracy_nb = metrics.accuracy_score(y_test, y_test_pred_nb)
print("Test set Accuracy (Naive Bayes): ", test_accuracy_nb)

# Print best parameters and best score from GridSearchCV for Naive Bayes
print("Best parameters for Naive Bayes: ", grid_search_nb.best_params_)
print("Best cross-validation score (accuracy) for Naive Bayes: ", grid_search_nb.best_score_)

# Define a function to calculate accuracy with tolerance
def calculate_accuracy_with_tolerance(true_values, predictions, tolerance):
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    return np.mean(np.abs(true_values - predictions) <= tolerance)

# Define tolerance values to test
tolerances = np.linspace(0.1, 5.0, 50)
accuracies_nb = [calculate_accuracy_with_tolerance(y_test, y_test_pred_nb, tol) for tol in tolerances]

# Plot REC Curve for Naive Bayes
plt.figure(figsize=(12, 8))
plt.plot(tolerances, accuracies_nb, color='green', lw=2, marker='o', label='Naive Bayes')
plt.xlabel('Tolerance')
plt.ylabel('Accuracy')
plt.title('Regression Error Characteristic (REC) Curve')
plt.grid(True)
plt.legend()
plt.show()

# Calculate REC AUC for Naive Bayes
rec_auc_nb = np.trapz(accuracies_nb, tolerances)
print(f"REC AUC for Naive Bayes: {rec_auc_nb:.4f}")
