import os
import cv2
import numpy as np

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_NAME = "Video Annotation"

SOURCE_DIR = "deadlift_processed"
SNAPSHOT_DIR = "deadlift_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def show_start_screen():
    """Displays a splash screen before annotation starts."""
    start_screen = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

    cv2.putText(
        start_screen,
        "Press any key to start...",
        (int(DISPLAY_WIDTH * 0.3), int(DISPLAY_HEIGHT * 0.5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, start_screen)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed


def save_snapshot(frame, video_name, frame_pos, phase=None):
    """Save a snapshot of the current frame with frame number and phase."""
    # Create video-specific subfolder
    video_snapshot_dir = os.path.join(SNAPSHOT_DIR, os.path.splitext(video_name)[0])
    os.makedirs(video_snapshot_dir, exist_ok=True)

    # Create filename with frame number and phase
    phase_str = f"_{phase}" if phase else ""
    filename = f"frame_{frame_pos}{phase_str}.jpg"
    filepath = os.path.join(video_snapshot_dir, filename)

    # Save the frame
    cv2.imwrite(filepath, frame)


def annotate_video(video_path):
    """Plays a video for annotation with rewind, fast-forward, and snapshot."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    video_name = os.path.basename(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    annotations = []
    last_key = None
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if the video ends

        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame

        # Resize and display the frame
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(10) & 0xFF  # Adjusted delay for smooth playback

        if key == ord(" "):  # Pause/Play
            paused = not paused

        elif key == ord("a"):  # Rewind 1 second
            frame_pos = max(0, frame_pos - fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        elif key == ord("d"):  # Fast-forward 1 second
            frame_pos += fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        elif key == ord("t") and last_key != "t":  # Top annotation
            annotations.append((frame_pos, "top"))
            save_snapshot(frame, video_name, frame_pos, "top")
            last_key = "t"

        elif key == ord("b") and last_key != "b":  # Bottom annotation
            annotations.append((frame_pos, "bottom"))
            save_snapshot(frame, video_name, frame_pos, "bottom")
            last_key = "b"

        elif key == ord("s"):  # Save snapshot without annotation
            save_snapshot(frame, video_name, frame_pos)
            print(f"Snapshot saved at frame {frame_pos}")

        elif key == ord("q"):  # Quit
            break

    cap.release()
    return annotations


# Get all video files
video_files = [
    os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(".mp4")
]

# Store annotations
all_annotations = {}

if video_files:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # Show splash screen
    show_start_screen()

current_video_index = 0
while current_video_index < len(video_files):
    video_file = video_files[current_video_index]

    annotations = annotate_video(video_file)
    if annotations:
        all_annotations[os.path.basename(video_file)] = annotations

    # After annotation, ask for next action
    action_screen = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        action_screen,
        f"Video {current_video_index + 1}/{len(video_files)} completed",
        (int(DISPLAY_WIDTH * 0.3), int(DISPLAY_HEIGHT * 0.4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        action_screen,
        "Press 'd' for next video, 'a' for previous video, 'q' to quit",
        (int(DISPLAY_WIDTH * 0.2), int(DISPLAY_HEIGHT * 0.5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, action_screen)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("d"):  # Next video
        current_video_index += 1
    elif key == ord("a") and current_video_index > 0:  # Previous video
        current_video_index -= 1
    elif key == ord("q"):  # Quit
        break      

cv2.destroyAllWindows()
print("\nAnnotation complete!")

"""
**Keys**:
- `a` for rewind / prev video
- `d` for fast-forward / next video
- `t` for top phase
- `b` for bottom phase
- `q` for quit
- `space` for play / pause
"""
