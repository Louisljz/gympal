import cv2
import mediapipe as mp
import numpy as np
import time
from inference import get_model

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# Configuration parameters
GRIP_THRESHOLD = 0.05  # Adjust this value based on testing
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White text
BG_COLOR = (0, 0, 0)  # Black background
BARBELL_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for barbell detection

# Leniency parameters
BARBELL_DISAPPEARANCE_TOLERANCE = 15  # Frames to allow barbell to disappear
GRIP_PERSISTENCE_TIME = 1.5  # Seconds to maintain grip state after barbell disappears

# Status tracking
status = "Initializing..."
is_gripping = False
grip_start_time = None
grip_duration = 0
grip_locked = False

# Leniency tracking
barbell_missing_frames = 0
last_barbell_center = None
last_barbell_detection_time = None


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def process_frame(image, barbell_detections):
    global status, is_gripping, grip_start_time, grip_duration, grip_locked
    global barbell_missing_frames, last_barbell_center, last_barbell_detection_time

    # Process the frame with MediaPipe Pose
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)

    # Create a copy of the image for visualization
    vis_image = image.copy()

    # Check if barbell is detected
    barbell_detected = False
    barbell_center = None

    for box in barbell_detections.predictions:
        if (
            box.confidence > BARBELL_CONFIDENCE_THRESHOLD
            and box.class_name == "barbell"
        ):
            barbell_detected = True

            # Get center of the barbell
            x, y, w, h = box.x, box.y, box.width, box.height
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            barbell_center = (int(x), int(y))

            # Reset missing frames counter and update last detection time
            barbell_missing_frames = 0
            last_barbell_center = barbell_center
            last_barbell_detection_time = time.time()

            # Draw bounding box for barbell
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with confidence
            label = f"Barbell: {box.confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                FONT,
                FONT_SCALE,
                (0, 255, 0),
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
            break

    # Apply leniency if barbell not detected but was recently seen
    if not barbell_detected and last_barbell_center is not None:
        current_time = time.time()
        time_since_last_detection = (
            current_time - last_barbell_detection_time
            if last_barbell_detection_time
            else float("inf")
        )

        # Check if we're within the tolerance window
        if (
            barbell_missing_frames < BARBELL_DISAPPEARANCE_TOLERANCE
            and time_since_last_detection < GRIP_PERSISTENCE_TIME
        ):
            barbell_missing_frames += 1
            barbell_center = last_barbell_center

            # Show interpolated barbell with different color to indicate it's estimated
            cv2.circle(
                vis_image, barbell_center, 10, (0, 165, 255), -1
            )  # Orange circle

            # Add indicator text
            cv2.putText(
                vis_image,
                f"Barbell tracking ({BARBELL_DISAPPEARANCE_TOLERANCE - barbell_missing_frames})",
                (barbell_center[0] - 100, barbell_center[1] - 20),
                FONT,
                FONT_SCALE,
                (0, 165, 255),
                FONT_THICKNESS,
            )

            barbell_detected = True  # Consider barbell as detected for logic purposes
        else:
            # Reset tracking when tolerance exceeded
            last_barbell_center = None
            last_barbell_detection_time = None

    # Check if pose is detected
    pose_detected = False
    wrist_points = []

    if pose_results.pose_landmarks:
        pose_detected = True

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            vis_image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2
            ),
        )

        # Get wrist landmarks (both left and right)
        landmarks = pose_results.pose_landmarks.landmark
        h, w, _ = image.shape

        # Left wrist
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_wrist_point = (int(left_wrist.x * w), int(left_wrist.y * h))
        wrist_points.append(left_wrist_point)

        # Right wrist
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_wrist_point = (int(right_wrist.x * w), int(right_wrist.y * h))
        wrist_points.append(right_wrist_point)

        # Highlight wrists
        for wrist_point in wrist_points:
            cv2.circle(vis_image, wrist_point, 8, (0, 0, 255), -1)

    # Add leniency indicators to status panel
    leniency_text = ""
    if barbell_missing_frames > 0:
        leniency_text = f"Barbell tracking: {BARBELL_DISAPPEARANCE_TOLERANCE - barbell_missing_frames} frames left"

    # Determine status
    if not barbell_detected and not pose_detected:
        status = "No barbell or person detected"
        is_gripping = False
        grip_locked = False
    elif not barbell_detected:
        status = "No barbell detected"
        is_gripping = False
        grip_locked = False
    elif not pose_detected:
        status = "No person detected"
        is_gripping = False
        grip_locked = False
    elif barbell_detected and pose_detected and barbell_center:
        # Calculate minimum distance between any wrist and the barbell
        min_distance = float("inf")
        closest_wrist = None

        for wrist_point in wrist_points:
            distance = calculate_distance(wrist_point, barbell_center)
            if distance < min_distance:
                min_distance = distance
                closest_wrist = wrist_point

        # Draw line between closest wrist and barbell
        if closest_wrist:
            cv2.line(vis_image, closest_wrist, barbell_center, (255, 0, 0), 2)

        # Normalize distance based on image size for better threshold consistency
        image_diagonal = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
        normalized_distance = min_distance / image_diagonal

        # Display distance
        distance_text = f"Distance: {normalized_distance:.3f}"
        cv2.putText(
            vis_image,
            distance_text,
            (10, 150),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

        # Check if gripping (when wrist is close enough to barbell)
        if normalized_distance < GRIP_THRESHOLD:
            if not is_gripping:
                is_gripping = True
                grip_start_time = time.time()
                status = "Gripping barbell"
            else:
                grip_duration = time.time() - grip_start_time

                # Check if grip is maintained for enough time to be considered "locked"
                if grip_duration > 1.0 and not grip_locked:
                    grip_locked = True
                    status = "LOCKED IN - Grip secured"
                else:
                    status = f"Gripping barbell ({grip_duration:.1f}s)"
        else:
            is_gripping = False
            grip_locked = False
            status = f"Not gripping (Distance: {normalized_distance:.3f})"

    # Display status on frame
    # Create a semi-transparent overlay for text background
    overlay = vis_image.copy()
    cv2.rectangle(
        overlay, (0, 0), (500, 210), BG_COLOR, -1
    )  # Made taller for extra leniency info
    cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)

    # Add status text
    cv2.putText(
        vis_image,
        f"Status: {status}",
        (10, 30),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )

    # Add detection information
    cv2.putText(
        vis_image,
        f"Barbell detected: {barbell_detected}",
        (10, 60),
        FONT,
        FONT_SCALE,
        (0, 255, 0) if barbell_detected else (0, 0, 255),
        FONT_THICKNESS,
    )

    # Add leniency information if active
    if leniency_text:
        cv2.putText(
            vis_image,
            leniency_text,
            (10, 180),
            FONT,
            FONT_SCALE,
            (0, 165, 255),  # Orange
            FONT_THICKNESS,
        )

    cv2.putText(
        vis_image,
        f"Person detected: {pose_detected}",
        (10, 90),
        FONT,
        FONT_SCALE,
        (0, 255, 0) if pose_detected else (0, 0, 255),
        FONT_THICKNESS,
    )

    # Display grip status
    if is_gripping:
        if grip_locked:
            color = (0, 255, 0)  # Green for locked
        else:
            color = (0, 255, 255)  # Yellow for gripping
    else:
        color = (0, 0, 255)  # Red for not gripping

    cv2.putText(
        vis_image,
        f"Grip: {'LOCKED' if grip_locked else 'ACTIVE' if is_gripping else 'NONE'}",
        (10, 120),
        FONT,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
    )

    return vis_image


def main(video_source=0, save_output=True, output_filename="annotated_output.mp4"):
    """
    Main function for barbell grip detection

    Args:
        video_source: Can be 0 for webcam, or a file path for a video
        save_output: Boolean to determine if output should be saved
        output_filename: Filename for the saved output video
    """
    # Load the barbell detection model
    model = get_model(
        model_id="barbell-object-detection/5", api_key="b2M97eRiEjdBuK9ZSdyG"
    )

    # Open video source (webcam or file)
    cap = cv2.VideoCapture(video_source)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer if saving is enabled
    video_writer = None
    if save_output:
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' for AVI
        video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, (frame_width, frame_height)
        )
        print(f"Output video will be saved to: {output_filename}")

    print("Starting barbell grip detection. Press 'q' to quit.")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error capturing frame")
                break

            # Run barbell detection inference
            barbell_results = model.infer(frame)[0]

            # Process frame with pose detection and grip analysis
            processed_frame = process_frame(frame, barbell_results)

            # Save the processed frame if saving is enabled
            if save_output and video_writer is not None:
                video_writer.write(processed_frame)

            # Display the resulting frame
            cv2.imshow("Barbell Grip Detection", processed_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up
        cap.release()
        if save_output and video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Application closed")


if __name__ == "__main__":
    main(
        "exercises/deadlift_test.mkv",  # Use video file
        save_output=True,  # Enable saving
        output_filename="exercises/deadlift_analysis.mp4",  # Output filename
    )
