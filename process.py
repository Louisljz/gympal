import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def process_video(input_path, output_path, target_duration=5, target_fps=24):
    """
    Process a video:
    1. Crop to target duration (first 5 seconds)
    2. Standardize to target FPS
    """
    try:
        # Load the video
        clip = VideoFileClip(input_path)

        # Crop to first 5 seconds
        if clip.duration > target_duration:
            clip = clip.subclip(0, target_duration)

        # Set the FPS
        clip = clip.set_fps(target_fps)

        # Write the processed video
        clip.write_videofile(output_path, codec="libx264", fps=target_fps)

        # Close the clip to release resources
        clip.close()

        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_all_videos(input_dir, output_dir):
    """Process all videos in the input directory"""
    # Get all video files
    video_files = [
        f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi", ".mov"))
    ]

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos to process")

    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)
        process_video(input_path, output_path)

    print("Video processing complete")


# Run the processing
input_dir = "exercises/deadlift"
output_dir = "deadlift_processed"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
process_all_videos(input_dir, output_dir)
