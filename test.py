import cv2
import mediapipe as mp
import joblib
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load the trained model
model = joblib.load("pose_classifier.pkl")


# Function to extract and filter keypoints
def extract_keypoints(results):
    keypoints = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        keypoints[mp_pose.PoseLandmark(idx).name] = (landmark.x, landmark.y)

    # Filter the required keypoints by landmark name
    selected_keypoints = []
    for landmark_name in [
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    ]:
        if landmark_name in keypoints:
            selected_keypoints.extend(keypoints[landmark_name])

    return selected_keypoints


# Read the image files
image1 = cv2.imread("snapshots/frame_58_top.jpg")
image2 = cv2.imread("snapshots/frame_117_bottom.jpg")

# Process the images
images = [image1, image2]
for idx, image in enumerate(images):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Extract and filter keypoints
        selected_keypoints = extract_keypoints(results)

        # Reshape for prediction
        keypoints_array = np.array(selected_keypoints).reshape(1, -1)

        # Predict the pose
        pose_label = model.predict(keypoints_array)[0]

        # Display the pose label on the image
        cv2.putText(
            image,
            pose_label,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Save the processed image
    cv2.imwrite(f"processed_image_{idx+1}.jpg", image)
