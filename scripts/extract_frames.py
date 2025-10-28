import cv2
import os
import sys

def extract_frames(video_path, num_frames, output_dir="output_frames"):
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    # Calculate step size
    step = max(1, total_frames // num_frames)

    count = 0
    saved = 0
    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Saved {saved} frames to '{output_dir}'.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <video.mp4> <num_frames>")
        sys.exit(1)

    video_file = sys.argv[1]
    num_frames = int(sys.argv[2])

    extract_frames(video_file, num_frames)
