import os
import cv2
import csv
import mediapipe as mp
from tqdm import tqdm

SOURCE_DIR = "deadlift_snapshots"
OUTPUT_DIR = "deadlift_keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.6,
)

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

csv_filename = "deadlift_keypoints.csv"
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)

header = ["video_file", "frame_no"]
for keypoint in KEYPOINTS.keys():
    header.extend([f"{keypoint}_x", f"{keypoint}_y", f"{keypoint}_z"])
header.append("label")  # Add label column
csv_writer.writerow(header)

for video_name in tqdm(os.listdir(SOURCE_DIR), desc="Detecting keypoints"):
    video_folder = os.path.join(SOURCE_DIR, video_name)

    # output_video_folder = os.path.join(OUTPUT_DIR, video_name)
    # os.makedirs(output_video_folder, exist_ok=True)

    for image_file in os.listdir(video_folder):
        parts = image_file.split("_")
        frame_no = int(parts[1])
        label = parts[2].split('.')[0]
        row = [video_name, frame_no]

        image_path = os.path.join(video_folder, image_file)
        image = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            for idx in KEYPOINTS.values():
                landmark = results.pose_landmarks.landmark[idx]
                row.extend([landmark.x, landmark.y, landmark.z])

            # mp_drawing.draw_landmarks(
            #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            # )
            # cv2.imwrite(os.path.join(output_video_folder, image_file), image)
        else:
            print(f"No pose landmarks detected in: {image_path}")
            continue

        row.append(label)
        csv_writer.writerow(row)
