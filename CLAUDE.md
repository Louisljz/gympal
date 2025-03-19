# GymPal Development Guidelines

## Project Overview
GymPal is a machine learning application for detecting and analyzing exercise form using computer vision and pose estimation.

## Environment Setup
```bash
pip install -r requirements.txt
```

## Run Commands
- Process videos: `python process.py`
- Create snapshots: `python snapshot.py`
- Extract keypoints: `python keypoint.py`
- Augment data: `python augment.py`
- Train model: `python train.py`
- Run inference: `python inference.py`

## Code Style Guidelines
- **Imports**: Group standard library imports first, followed by third-party packages, then local modules
- **Formatting**: Use 4 spaces for indentation
- **Naming**: Use snake_case for variables/functions and PascalCase for classes
- **Constants**: Define constants at the top of files in UPPERCASE
- **Error Handling**: Use try/except blocks with specific exception types
- **Comments**: Use docstrings for functions, classes, and modules
- **Type hints**: Not currently used but encouraged for new code

## Directory Structure
- `/exercises`: Raw exercise videos
- `/deadlift_processed`: Processed videos
- `/deadlift_snapshots`: Frame snapshots with annotations
- `/deadlift_keypoints`: Extracted pose keypoints data