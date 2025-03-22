import cv2
from inference import get_model


def main(video_file):
    # Load the model
    model = get_model(
        model_id="barbell-object-detection/5", api_key="b2M97eRiEjdBuK9ZSdyG"
    )

    # Open video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    print("Video opened successfully. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image or end of video reached")
            break

        # Run inference on the frame
        results = model.infer(frame)[0]

        for box in results.predictions:
            if box.confidence > 0.8:
                x, y, w, h = box.x, box.y, box.width, box.height
                x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = f"{box.class_name}: {box.confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Display the resulting frame
        cv2.imshow("Barbell Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("exercises/deadlift/deadlift_1.mp4")
