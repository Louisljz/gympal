import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

source_path = "keypoint_data/"
output = {}

for key_file in os.listdir(source_path):
    if not key_file.endswith(".json"):
        continue
    key_file_path = os.path.join(source_path, key_file)
    label = key_file.split(".")[0].split("_")[-1]
    with open(key_file_path, "r") as f:
        data = json.load(f)

    dimensions = data['model_predictions']['image']
    width, height = dimensions['width'], dimensions['height']

    keypoints = data["model_predictions"]["predictions"][0]["keypoints"]
    for keypoint in keypoints:
        point_x = keypoint["class"] + "_x"
        point_y = keypoint["class"] + "_y"
        if point_x not in output:
            output[point_x] = []
            output[point_y] = []

        norm_x, norm_y = keypoint["x"] / width, keypoint["y"] / height
        output[point_x].append(norm_x)
        output[point_y].append(norm_y)

    if "pose" not in output:
        output["pose"] = []
    output["pose"].append(label)

df = pd.DataFrame(output)
df.drop(
    [
        "nose_x",
        "nose_y",
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
        "left_ear_x",
        "left_ear_y",
        "right_ear_x",
        "right_ear_y",
    ],
    inplace=True,
    axis=1,
)
print(df.columns)
df.to_csv("keypoint_data/keypoint_data.csv", index=False)

# Extract features and labels
X = df.iloc[:, :-1].values  # All columns except the last one (pose label)
y = df.iloc[:, -1].values  # Last column (pose label)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
import joblib

joblib.dump(clf, "pose_classifier.pkl")
print("Model saved as 'pose_classifier.pkl'")
