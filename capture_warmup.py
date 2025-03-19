import cv2
import os
import numpy as np

# Constants
ACTIVITIES = ["walking", "sitting", "stretching", "drinking_water", "checking_phone"]
FRAMES_PER_ACTIVITY = 10
CAPTURE_FPS = 5  # Capture at 5 FPS
OUTPUT_DIR = "exercises/warmup"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def draw_text(image, text, font_scale=1, thickness=2, color=(0, 0, 255)):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[
        0
    ]
    text_x = (DISPLAY_WIDTH - text_size[0]) // 2
    text_y = (DISPLAY_HEIGHT + text_size[1]) // 2
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def show_splash_screen(text):
    splash = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8
    )  # Black background
    draw_text(splash, text)
    cv2.imshow("Capture Warmup Activities", splash)
    key = cv2.waitKey(0) & 0xFF  # Wait for a key press
    if key == ord("q"):
        return False  # Exit if 'q' is pressed

    # Countdown
    for count in ["3", "2", "1"]:
        splash.fill(0)  # Clear screen
        draw_text(splash, count, font_scale=2, thickness=3)
        cv2.imshow("Capture Warmup Activities", splash)
        cv2.waitKey(1000)  # Wait 1 second per count

    return True


def capture_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    try:
        for activity in ACTIVITIES:
            if not show_splash_screen(
                f"Next: {activity.upper()} Press any key to continue"
            ):
                print("Capture aborted by user.")
                return  # Exit if user pressed 'q'

            for i in range(FRAMES_PER_ACTIVITY):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                filename = f"{OUTPUT_DIR}/{activity}_{i+1}.jpg"
                cv2.imwrite(filename, frame)

                # Display frame with activity info
                cv2.putText(
                    frame,
                    f"Activity: {activity} ({i+1}/{FRAMES_PER_ACTIVITY})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Capture Warmup Activities", frame)

                # Wait to maintain FPS, allow quitting
                if cv2.waitKey(int(1000 / CAPTURE_FPS)) & 0xFF == ord("q"):
                    print("Capture interrupted by user")
                    return

            print(f"Completed {activity}")

        print("\nCapture complete! All activities recorded.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting warmup activity capture")
    capture_frames()
