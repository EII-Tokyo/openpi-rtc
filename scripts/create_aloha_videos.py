#!/usr/bin/env python3
"""
Script to create videos from BiPlay dataset's aloha_pen_uncap_diverse_raw folder.
Each video will show four camera views (cam_high, cam_low, cam_left_wrist, cam_right_wrist) 
arranged in a 2x2 grid layout.

Usage:
    python scripts/create_aloha_videos.py --output-dir ./videos
"""

import argparse
import cv2
import h5py
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tqdm
import multiprocessing as mp
from functools import partial
from huggingface_hub import snapshot_download
import tempfile
import glob


def download_and_get_hdf5_paths(repo_id: str = "oier-mees/BiPlay",
                               subset: str = "aloha_pen_uncap_diverse_raw",
                               cache_dir: Optional[str] = None) -> List[str]:
    """
    Download dataset from HuggingFace and get paths to HDF5 files.
    
    Args:
        repo_id: HuggingFace repository ID
        subset: Subset name containing HDF5 files
        cache_dir: Directory to cache downloaded files
        
    Returns:
        List of paths to HDF5 files
    """
    print(f"Downloading repository {repo_id}")
    
    # Download the entire repository directly
    repo_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        resume_download=True,
        repo_type="dataset"
    )
    
    print(f"Repository downloaded to: {repo_dir}")
    
    # Recursively find all HDF5 files in the repository
    repo_path = Path(repo_dir)
    hdf5_files = list(repo_path.rglob("*.hdf5"))
    
    # Filter files that contain the subset name in their path
    if subset:
        hdf5_files = [f for f in hdf5_files if subset in str(f)]
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Print found files for debugging
    for f in hdf5_files:
        print(f"  - {f}")
    
    return [str(f) for f in hdf5_files]


def create_2x2_grid(images: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create a 2x2 grid layout from four camera images.
    
    Args:
        images: Dictionary containing camera images with keys:
               'cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'
    
    Returns:
        Combined image in 2x2 grid layout
    """
    # Define the layout: [top_left, top_right, bottom_left, bottom_right]
    layout = [
        ('cam_high', 'cam_low'),
        ('cam_left_wrist', 'cam_right_wrist')
    ]
    
    # Get image dimensions from the first image
    first_img = next(iter(images.values()))
    h, w = first_img.shape[:2]
    
    # Create the combined image
    combined_h = h * 2
    combined_w = w * 2
    combined_img = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    
    # Place images in the grid
    for row_idx, row in enumerate(layout):
        for col_idx, cam_name in enumerate(row):
            if cam_name in images:
                img = images[cam_name]
                # Ensure image is in RGB format
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                
                # Calculate position in the grid
                y_start = row_idx * h
                y_end = (row_idx + 1) * h
                x_start = col_idx * w
                x_end = (col_idx + 1) * w
                combined_img[y_start:y_end, x_start:x_end] = img

    return combined_img


def process_hdf5_file(args: Tuple[str, str, int]) -> str:
    """Process a single HDF5 file and create a video.
    
    Args:
        args: Tuple containing (input_file_path, output_dir, fps)
    
    Returns:
        Path to the created video file
    """
    input_file, output_dir, fps = args
    try:
        # Create output filename from input filename
        input_path = Path(input_file)
        output_path = Path(output_dir) / f"{input_path.stem}.mp4"
        
        print(f"Processing {input_path.name}")
        
        with h5py.File(input_file, 'r') as f:
            # Debug: print all HDF5 structure
            def print_structure(name, obj):
                print(f"{name}: {type(obj).__name__}")
                if hasattr(obj, 'shape'):
                    print(f"  shape: {obj.shape}")
                if hasattr(obj, 'dtype'):
                    print(f"  dtype: {obj.dtype}")
            
            print(f"=== HDF5 file structure for {input_path.name} ===")
            f.visititems(print_structure)
            print("=" * 50)
            
            # Try different possible structures
            possible_paths = [
                'observations/images/cam_high',
                'obs/images/cam_high', 
                'cam_high',
                'images/cam_high'
            ]
            
            frames_dataset = None
            for path in possible_paths:
                try:
                    frames_dataset = f[path]
                    print(f"Found frames at: {path}")
                    break
                except KeyError:
                    continue
            
            if frames_dataset is None:
                print(f"Could not find camera data in {input_file}")
                return ""
            # Get number of frames from dataset shape
            num_frames = frames_dataset.shape[0]
            print(f"Number of frames: {num_frames}")
            
            # Get first frame to determine dimensions
            first_frame_flat = {
                'cam_high': frames_dataset[0],
                'cam_low': f[path.replace('cam_high', 'cam_low')][0],
                'cam_left_wrist': f[path.replace('cam_high', 'cam_left_wrist')][0],
                'cam_right_wrist': f[path.replace('cam_high', 'cam_right_wrist')][0]
            }
            
            # Decode JPEG compressed images
            first_frame = {}
            for key, data in first_frame_flat.items():
                # Data is JPEG compressed, decode it
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    first_frame[key] = img
                else:
                    print(f"Failed to decode image for {key}")
                    return ""
            
            h, w = first_frame['cam_high'].shape[:2]
            video_h = h * 2
            video_w = w * 2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))
            
            # Process all frames
            for frame_idx in tqdm.tqdm(range(num_frames), desc=f"Processing {input_path.name}", leave=False):
                images_flat = {
                    'cam_high': frames_dataset[frame_idx],
                    'cam_low': f[path.replace('cam_high', 'cam_low')][frame_idx],
                    'cam_left_wrist': f[path.replace('cam_high', 'cam_left_wrist')][frame_idx],
                    'cam_right_wrist': f[path.replace('cam_high', 'cam_right_wrist')][frame_idx]
                }
                
                # Decode JPEG compressed images  
                images = {}
                for key, data in images_flat.items():
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is not None:
                        images[key] = img
                    else:
                        print(f"Failed to decode image for {key} at frame {frame_idx}")
                        continue
                
                # Create 2x2 grid
                combined_img = create_2x2_grid(images)
                
                # Write frame to video
                video_writer.write(combined_img)
            
            video_writer.release()
            
        return str(output_path)
        
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return ""


def main():
    parser = argparse.ArgumentParser(description="Create videos from BiPlay dataset")
    parser.add_argument("--output-dir", default="./videos",
                       help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for output videos")
    parser.add_argument("--num-processes", type=int, default=None,
                       help="Number of processes to use. Defaults to CPU count.")
    parser.add_argument("--repo-id", default="oier-mees/BiPlay",
                       help="HuggingFace repository ID")
    parser.add_argument("--subset", default="aloha_pen_uncap_diverse_raw",
                       help="Dataset subset containing HDF5 files")
    parser.add_argument("--cache-dir", default=None,
                       help="Directory to cache downloaded files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset and get HDF5 file paths
    hdf5_files = download_and_get_hdf5_paths(args.repo_id, args.subset, args.cache_dir)
    
    if not hdf5_files:
        print(f"No HDF5 files found in repository {args.repo_id}, subset {args.subset}")
        return 1
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Set up multiprocessing
    num_processes = args.num_processes or mp.cpu_count()
    print(f"Using {num_processes} processes")
    
    # Prepare arguments for each process
    process_args = [(f, args.output_dir, args.fps) for f in hdf5_files]
    
    # Process files in parallel
    with mp.Pool(num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_hdf5_file, process_args),
            total=len(process_args),
            desc="Processing files"
        ))
    
    # Print results
    successful = [r for r in results if r]
    print(f"\nProcessed {len(successful)} videos successfully")
    print(f"Videos saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    exit(main()) 