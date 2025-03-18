import os
import cv2
from tqdm import tqdm

def extract_frames_around_snapshots(source_dir, snapshots_dir, frame_range=2):
    """
    Extract frames around the snapshot frames from processed videos and save them
    to the same snapshot directory structure.
    
    Args:
        source_dir: Directory containing processed videos
        snapshots_dir: Directory containing snapshots organized by video name
        frame_range: Number of frames to extract before and after the snapshot frame
    """
    # Get all video files
    video_files = [f for f in os.listdir(source_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    
    if not video_files:
        print(f"No video files found in {source_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(source_dir, video_file)
        
        # Check if this video has snapshots
        video_snapshot_dir = os.path.join(snapshots_dir, video_name)
        if not os.path.exists(video_snapshot_dir):
            print(f"No snapshots found for {video_name}, skipping...")
            continue
        
        # Get all snapshot files for this video
        snapshot_files = [f for f in os.listdir(video_snapshot_dir) if f.endswith(".jpg")]
        
        if not snapshot_files:
            print(f"No snapshot files found for {video_name}, skipping...")
            continue
        
        # Extract frame numbers and phases from snapshot filenames
        frame_info = []
        for snapshot_file in snapshot_files:
            # Example: frame_54_top.jpg
            try:
                parts = snapshot_file.split('_')
                if len(parts) == 3 and parts[0] == "frame":
                    frame_no = int(parts[1])
                    phase = parts[2].split('.')[0]  # Get 'top' or 'bottom' without extension
                    frame_info.append((frame_no, phase))
            except (ValueError, IndexError):
                print(f"Could not parse frame info from {snapshot_file}, skipping...")
                continue
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open {video_path}, skipping...")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_no, phase in frame_info:
            frames_to_extract = []
            for offset in range(-frame_range, frame_range + 1):
                # Skip the target frame (offset 0)
                if offset == 0:
                    continue
                
                target_frame = frame_no + offset
                if 0 <= target_frame < total_frames:
                    frames_to_extract.append(target_frame)
            
            # Extract only the specific frames
            for f in frames_to_extract:
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Could not read frame {f} from {video_path}, skipping...")
                    continue
                
                # Save the frame to the same snapshot directory with phase information
                frame_filename = f"frame_{f}_{phase}.jpg"
                output_path = os.path.join(video_snapshot_dir, frame_filename)
                cv2.imwrite(output_path, frame)
        
        # Release the video
        cap.release()
    
    print("Frame extraction complete!")

# Run the augmentation
source_dir = "deadlift_processed"
snapshots_dir = "deadlift_snapshots"

extract_frames_around_snapshots(source_dir, snapshots_dir)
