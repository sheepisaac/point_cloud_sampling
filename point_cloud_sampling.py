import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import sys
import time

def log(message, overwrite=False):
    """Log a message to the console, optionally overwriting the last line."""
    if overwrite:
        sys.stdout.write(f"\r{message} ")
        sys.stdout.flush()
    else:
        print(message, flush=True)

def load_point_cloud(file_path):
    """Load a point cloud from a file."""
    log(f"Loading point cloud from {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def save_point_cloud(points, output_dir, input_filename, k):
    """Save a point cloud to a file with a modified name."""
    base_name = os.path.splitext(input_filename)[0]  # Remove extension
    output_filename = f"{base_name}_sampled_k{k}.ply"
    output_path = os.path.join(output_dir, output_filename)
    
    log(f"Saving sampled point cloud to {output_path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_path, pcd)

def estimate_processing_time(num_points, k):
    """Estimate processing time based on point cloud size and k value."""
    estimated_time = (num_points / 100000) * (k / 10) * 5  # Rough heuristic estimation
    log(f"Estimated sampled point cloud size: {num_points // k} points")
    log(f"Estimated processing time: {estimated_time:.2f} seconds")

def sample_point_cloud(points, k):
    """Perform sampling on the point cloud using k-nearest neighbors."""
    log(f"Performing sampling with k={k} neighbors")
    tree = KDTree(points)  # Create a KD-tree for efficient neighbor search
    sampled_points = []
    
    start_time = time.time()
    for i, point in enumerate(points):
        _, idx = tree.query(point, k=k)  # Find k nearest neighbors
        sampled_point = np.mean(points[idx], axis=0)  # Compute mean position
        sampled_points.append(sampled_point)
        if i % 1000 == 0:
            elapsed_time = time.time() - start_time
            log(f"Processed {i} points... (Elapsed: {elapsed_time:.2f}s)", overwrite=True)
    
    log("")  # Move to a new line after progress output
    return np.array(sampled_points)

def process_input(input_path, output_dir, k):
    """Process a single file or all point cloud files in a directory."""
    if os.path.isfile(input_path):
        # Single file processing
        file_name = os.path.basename(input_path)
        log(f"Processing single file: {file_name}")
        points = load_point_cloud(input_path)
        estimate_processing_time(len(points), k)
        sampled_points = sample_point_cloud(points, k)
        save_point_cloud(sampled_points, output_dir, file_name, k)
        log(f"Completed processing: {file_name}")
    elif os.path.isdir(input_path):
        # Directory processing
        log(f"Processing directory: {input_path}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create output directory if it doesn't exist
        
        for file_name in os.listdir(input_path):
            if file_name.endswith('.ply') or file_name.endswith('.pcd'):
                log(f"Processing file: {file_name}")
                input_file = os.path.join(input_path, file_name)
                points = load_point_cloud(input_file)
                estimate_processing_time(len(points), k)
                sampled_points = sample_point_cloud(points, k)
                save_point_cloud(sampled_points, output_dir, file_name, k)
                log(f"Completed processing: {file_name}")
    else:
        log(f"Error: {input_path} is not a valid file or directory.")

def main():
    """Parse command line arguments and initiate processing."""
    parser = argparse.ArgumentParser(description='Point Cloud Sampling')
    parser.add_argument('-ip', '--input_path', required=True, help='Input file or directory containing point cloud files')
    parser.add_argument('-op', '--output_path', required=True, help='Output directory for sampled point cloud files')
    parser.add_argument('-k', '--neighbors', type=int, required=True, help='Number of neighbors to use for sampling')
    
    args = parser.parse_args()
    log(f"Starting point cloud sampling with k={args.neighbors}")
    process_input(args.input_path, args.output_path, args.neighbors)
    log("Processing complete.")

if __name__ == "__main__":
    main()
