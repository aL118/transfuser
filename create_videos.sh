#!/bin/bash

# Script to convert PNG frames in each folder to MP4 videos
# Usage: ./create_videos.sh

INPUT_DIR="/fs/nexus-scratch/aliu1237/transfuser/debug_output_quadtree"
OUTPUT_DIR="/fs/nexus-scratch/aliu1237/transfuser/quadtree_videos"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed or not in PATH"
    exit 1
fi

echo "Converting PNG frames to MP4 videos..."

# Process each folder
for folder in "$INPUT_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        echo "Processing folder: $folder_name"

        # Check if folder contains PNG files
        png_count=$(ls "$folder"/*.png 2>/dev/null | wc -l)
        if [ "$png_count" -eq 0 ]; then
            echo "  Warning: No PNG files found in $folder_name, skipping..."
            continue
        fi

        echo "  Found $png_count PNG files"

        # Get the first PNG file to determine naming pattern
        first_png=$(ls "$folder"/*.png | head -1)
        first_num=$(basename "$first_png" .png)

        # Create MP4 with folder name as title
        output_file="$OUTPUT_DIR/${folder_name}.mp4"

        # Use ffmpeg to create video from PNG sequence
        # -framerate 10: 10 fps (adjust as needed)
        # -i: input pattern
        # -c:v libx264: H.264 codec
        # -pix_fmt yuv420p: pixel format for compatibility
        # -y: overwrite output file if exists
        ffmpeg -framerate 10 \
               -pattern_type glob \
               -i "$folder/*.png" \
               -c:v libx264 \
               -pix_fmt yuv420p \
               -y \
               "$output_file" 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "  ✓ Created: $output_file"
        else
            echo "  ✗ Failed to create video for $folder_name"
        fi
    fi
done

echo "Video conversion complete!"
echo "Videos saved to: $OUTPUT_DIR"