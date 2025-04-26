#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Maker for CLFT Conference

This script creates a compressed video from segmentation overlay images
in the output directory.
"""

import os
import cv2
import argparse
import glob
import re
from tqdm import tqdm

def natural_sort_key(s):
    """
    Sort strings that contain numbers in a natural way.
    E.g. ["img_1.png", "img_2.png", "img_10.png"] instead of
    ["img_1.png", "img_10.png", "img_2.png"]
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def create_video(input_dir, output_file, fps=10, codec='mp4v', quality=95, scale=1):
    """
    Creates a video from all images in the specified directory.
    
    Args:
        input_dir (str): Directory containing images
        output_file (str): Path to save the output video
        fps (int): Frames per second for the output video
        codec (str): FourCC codec code (e.g., 'mp4v', 'XVID', 'H264')
        quality (int): Quality of the video (0-100), higher is better quality
        scale (float): Scale factor to resize images (1.0 = original size)
    """
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return False
    
    # Get all image files (png, jpg, jpeg)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"Error: No image files found in '{input_dir}'.")
        return False
    
    # Sort the files naturally (so that image_1, image_2, ..., image_10 are in correct order)
    image_files.sort(key=natural_sort_key)
    
    # Read the first image to get dimensions
    img = cv2.imread(image_files[0])
    if img is None:
        print(f"Error: Could not read the first image '{image_files[0]}'.")
        return False
    
    # Resize if scale is not 1.0
    if scale != 1.0:
        height, width = int(img.shape[0] * scale), int(img.shape[1] * scale)
        img = cv2.resize(img, (width, height))
    else:
        height, width = img.shape[0], img.shape[1]
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_file))
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Check if directory is writable
        test_file = os.path.join(output_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("Output directory is writable")
    except Exception as e:
        print(f"Error with output directory: {str(e)}")
        print("Trying to save to current directory instead")
        output_file = os.path.basename(output_file)
        print(f"New output path: {output_file}")
    
    # Create a VideoWriter object - Try different codecs if in Docker
    # In Docker, some codecs may not be available with OpenCV
    available_codecs = ['mp4v', 'XVID', 'X264']
    used_codec = codec
    video = None
    
    for try_codec in [codec] + [c for c in available_codecs if c != codec]:
        try:
            fourcc = cv2.VideoWriter_fourcc(*try_codec)
            video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            if video.isOpened():
                used_codec = try_codec
                print(f"Using codec: {used_codec}")
                break
        except Exception as e:
            print(f"Failed to use codec {try_codec}: {str(e)}")
    
    if video is None or not video.isOpened():
        print("Error: Could not initialize VideoWriter with any codec.")
        return False
    
    # Process each image and add it to the video
    print(f"Creating video from {len(image_files)} images...")
    frame_count = 0
    try:
        for img_file in tqdm(image_files):
            img = cv2.imread(img_file)
            
            # Skip if image couldn't be read
            if img is None:
                print(f"Warning: Could not read image '{img_file}', skipping.")
                continue
                
            # Resize if needed
            if scale != 1.0:
                img = cv2.resize(img, (width, height))
            
            # Add just the filename to the image (no folder path)
            filename = os.path.basename(img_file)
            display_text = filename
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35  # Reduced from 0.5 to make text smaller
            font_thickness = 1
            text_color = (255, 255, 255)  # White text
            
            # Create a dark background for the text to make it more visible
            text_size = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
            text_x = 10  # Padding from left
            text_y = height - 10  # Padding from bottom
            
            # Draw a black rectangle with opacity
            overlay = img.copy()
            cv2.rectangle(overlay, 
                        (text_x - 5, text_y - text_size[1] - 5), 
                        (text_x + text_size[0] + 5, text_y + 5), 
                        (0, 0, 0), 
                        -1)  # Filled rectangle
            
            # Apply the overlay with transparency
            alpha = 0.7  # Higher opacity for better readability
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
            # Draw the text
            cv2.putText(img, display_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                
            # Write the frame to the video
            video.write(img)
            frame_count += 1
            
            # Every 1000 frames, check if we're still able to write
            if frame_count % 1000 == 0:
                if not video.isOpened():
                    print(f"Error: VideoWriter closed unexpectedly after {frame_count} frames")
                    return False
    except Exception as e:
        print(f"Error during video creation: {str(e)}")
        return False
    
    # Release the video writer
    video.release()
    
    # Check if the video was created successfully
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Video created successfully: {output_file}")
        print(f"Video size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print(f"Frames written: {frame_count}")
        return True
    else:
        print(f"Error: Failed to create video. File exists: {os.path.exists(output_file)}")
        print(f"Current working directory: {os.getcwd()}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create a video from segmentation overlay images.')
    parser.add_argument('--input_dir', type=str, default='output/clft_seg_results/overlay',
                        help='Directory containing the images (default: output/clft_seg_results/overlay)')
    parser.add_argument('--output_file', type=str, default='output/segmentation_video.mp4',
                        help='Output video file path (default: output/segmentation_video.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='Video codec (default: mp4v)')
    parser.add_argument('--quality', type=int, default=95,
                        help='Video quality 0-100 (default: 95)')
    parser.add_argument('--scale', type=float, default=1.5,
                        help='Scale factor for resizing images (default: 1.5)')
    parser.add_argument('--crf', type=int, default=23,
                        help='CRF value for H.264 compression (default: 23, lower means better quality)')
    
    args = parser.parse_args()
    
    # Check for Docker environment
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
    if in_docker:
        print("Detected Docker environment")
        # In Docker, ensure paths are absolute
        if not os.path.isabs(args.input_dir):
            args.input_dir = os.path.join('/workspace', args.input_dir)
        if not os.path.isabs(args.output_file):
            args.output_file = os.path.join('/workspace', args.output_file)
    
    # Handle cases where overlay directory might be in different locations
    if not os.path.exists(args.input_dir):
        alternate_paths = [
            'output/clft_seg_results/overlay',
            '/workspace/output/clft_seg_results/overlay',
            './output/clft_seg_results/overlay',
            os.path.join(os.getcwd(), 'output/clft_seg_results/overlay')
        ]
        
        for alt_path in alternate_paths:
            if os.path.exists(alt_path) and alt_path != args.input_dir:
                print(f"Input directory '{args.input_dir}' not found, using '{alt_path}' instead")
                args.input_dir = alt_path
                break
                
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    
    # Try using a variety of codec/container combinations that work well in Docker
    codecs_to_try = []
    
    # Default codec
    codecs_to_try.append((args.codec, args.output_file))
    
    # Fallback codecs and containers if the primary one fails
    base_name = os.path.splitext(args.output_file)[0]
    codecs_to_try.extend([
        ('XVID', f"{base_name}.avi"),  # AVI container with XVID codec (most compatible)
        ('X264', f"{base_name}_x264.mp4"),  # MP4 with X264
        ('MJPG', f"{base_name}.avi"),  # Motion JPEG
    ])
    
    # Try each codec until one works
    for codec, output_file in codecs_to_try:
        print(f"\nAttempting video creation with codec {codec} -> {output_file}")
        if create_video(args.input_dir, output_file, args.fps, codec, args.quality, args.scale):
            print(f"Successfully created video with codec {codec}")
            
            # We successfully created a video, so we can exit
            # Don't attempt to create a second "final" video
            return
            
    print("\nERROR: Failed to create video with any codec.")
    print("Try manually running ffmpeg to combine your images:")
    print("ffmpeg -framerate 10 -pattern_type glob -i 'output/clft_seg_results/overlay/*.png' -c:v libx264 -pix_fmt yuv420p output/segmentation_video.mp4")

if __name__ == "__main__":
    main()