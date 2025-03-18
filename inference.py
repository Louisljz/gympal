import mediapipe as mp
import numpy as np
import cv2
import pickle


class_map = {0: 'bottom', 1: 'top'}
with open("deadlift_classifier_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("deadlift_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)


mp_utils = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

KEYPOINTS = {
    "shoulder_left": 11,
    "shoulder_right": 12,
    "elbow_left": 13,
    "elbow_right": 14,
    "hip_left": 23,
    "hip_right": 24,
    "knee_left": 25,
    "knee_right": 26,
    "ankle_left": 27,
    "ankle_right": 28,
}
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

reps = 0
prev_phase = None
cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    phase_text = "-"
    if results.pose_landmarks:
        input = []
        for idx in KEYPOINTS.values():
            landmark = results.pose_landmarks.landmark[idx]
            input.extend([landmark.x, landmark.y, landmark.z])

        input = scaler.transform([input])
        output = model.predict_proba(input)[0]

        idx = np.argmax(output)
        prob = output[idx]

        if prob > 0.7: # threshold to configure
            curr_phase = class_map[idx]
            phase_text = curr_phase
            if prev_phase == "bottom" and curr_phase == "top":
                reps += 1
            prev_phase = curr_phase

        mp_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.putText(img, f"Phase: {phase_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(img, f"Reps: {reps}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Deadlift Counter', img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
