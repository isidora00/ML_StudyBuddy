from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, model_selection, metrics
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)
df = wine_quality.data

# Split the data into training, development, and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(df.features, df.targets, test_size=0.25, random_state=42, stratify=df.targets)
X_train, X_dev, y_train, y_dev = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Convert target variables to numpy arrays and flatten them
y_train = np.array(y_train).ravel()
y_dev = np.array(y_dev).ravel()
y_test = np.array(y_test).ravel()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, param_grid, name):
    pipeline = Pipeline([
        ('pca', PCA()),
        ('regressor', model)
    ])
    
    grid_search = model_selection.GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    y_dev_pred = best_model.predict(X_dev_scaled)
    dev_mse = metrics.mean_squared_error(y_dev, y_dev_pred)
    
    # Retrain on combined train and dev sets
    X_combined = np.concatenate([X_train_scaled, X_dev_scaled], axis=0)
    y_combined = np.concatenate([y_train, y_dev], axis=0)
    best_model.fit(X_combined, y_combined)
    
    # Test the model
    y_test_pred = best_model.predict(X_test_scaled)
    test_mse = metrics.mean_squared_error(y_test, y_test_pred)
    
    # Calculate REC curve
    tolerances = np.linspace(0.1, 5.0, 50)
    accuracies = [calculate_accuracy_with_tolerance(y_test, y_test_pred, tol) for tol in tolerances]
    
    return tolerances, accuracies, grid_search.best_params_, test_mse

# Define a function to calculate accuracy with tolerance
def calculate_accuracy_with_tolerance(true_values, predictions, tolerance):
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    return np.mean(np.abs(true_values - predictions) <= tolerance)

# Define models and parameter grids
models = {
    'Linear Regression': (linear_model.LinearRegression(), {'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}),
    'Random Forest': (RandomForestRegressor(), {
        'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }),
    'AdaBoost': (AdaBoostRegressor(), {
        'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 1.0]
    }),
    'SVR': (SVR(), {
        'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'regressor__C': [0.1, 1, 10, 100],
        'regressor__epsilon': [0.01, 0.1, 0.2],
        'regressor__gamma': ['scale', 'auto']
    }),
    'Naive Bayes': (GaussianNB(), {
        'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    })
}

# Plot REC Curve for all models
plt.figure(figsize=(12, 8))

for name, (model, param_grid) in models.items():
    tolerances, accuracies, best_params, test_mse = train_and_evaluate_model(model, param_grid, name)
    plt.plot(tolerances, accuracies, lw=2, marker='o', label=name)
    print(f"{name} - Best parameters: {best_params}")
    print(f"{name} - Test set Mean Squared Error: {test_mse:.4f}")

plt.xlabel('Tolerance')
plt.ylabel('Accuracy')
plt.title('Regression Error Characteristic (REC) Curve')
plt.grid(True)
plt.legend()
plt.show()
