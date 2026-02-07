#!/usr/bin/env python3

import os
import cv2
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_video_from_frames(input_folder, output_path, fps=10):
    """
    Create MP4 video from PNG frames in a folder

    Args:
        input_folder: Path to folder containing PNG frames
        output_path: Path for output MP4 file
        fps: Frames per second for output video
    """
    # Get all PNG files and sort them numerically
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    if not png_files:
        return False

    # Sort by numeric value of filename
    png_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Read first image to get dimensions
    first_frame = cv2.imread(png_files[0])
    if first_frame is None:
        return False

    height, width, layers = first_frame.shape

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        return False

    # Write frames to video with progress bar
    folder_name = os.path.basename(os.path.dirname(output_path))
    with tqdm(png_files, desc=f"Processing {folder_name}", unit="frame") as pbar:
        for png_file in pbar:
            frame = cv2.imread(png_file)
            if frame is None:
                continue

            video_writer.write(frame)

    # Release video writer
    video_writer.release()
    return True

def main():
    """Main function to process all folders"""
    input_dir = "/fs/nexus-scratch/aliu1237/transfuser/debug_output"
    output_dir = "/fs/nexus-scratch/aliu1237/transfuser/baseline_videos"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each folder
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    folders.sort()

    success_count = 0

    for folder_name in folders:
        if folder_name != 'scenario_6':
            continue
        print("Processing folder:", folder_name)
        folder_path = os.path.join(input_dir, folder_name)
        output_path = os.path.join(output_dir, f"{folder_name}.mp4")

        if create_video_from_frames(folder_path, output_path):
            success_count += 1

    print(f"\nSuccessfully created {success_count}/{len(folders)} videos in {output_dir}")

if __name__ == "__main__":
    main()