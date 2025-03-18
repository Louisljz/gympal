import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("deadlift_keypoints.csv")

# Drop non-numeric and unnecessary columns
X = data.iloc[:, 2:-1]  # Exclude video_file, frame_no, and label
y = data["label"].map({"top": 1, "bottom": 0})  # Encode labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit StandardScaler only on the training set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models for grid search
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
}

# Define parameter grids
param_grids = {
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "KNN": {"n_neighbors": [3, 5, 7]},
}

# Perform grid search with cross-validation
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(
        model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best {name} model: {grid_search.best_params_}")

# Find the best performing model by comparing test accuracy scores
best_model_name = max(best_models, key=lambda k: best_models[k].score(X_test, y_test))
best_model = best_models[best_model_name]
print(f"Selected best model: {best_model_name}")

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model to disk
model_filename = 'deadlift_classifier_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

scaler_filename = "deadlift_classifier_scaler.pkl"
with open(scaler_filename, "wb") as file:
    pickle.dump(scaler, file)
print(f"Model saved to {model_filename}")
