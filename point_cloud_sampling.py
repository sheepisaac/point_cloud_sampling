import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
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
    """Load a point cloud from a file, including colors if available."""
    log(f"Loading point cloud from {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points), (np.asarray(pcd.colors) if pcd.has_colors() else None)

def save_point_cloud(points, colors, output_dir, input_filename, k):
    """Save a point cloud with color information to a file."""
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_sampled_k{k}.ply"
    output_path = os.path.join(output_dir, output_filename)
    
    log(f"Saving sampled point cloud to {output_path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.clip(colors, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

def sample_point_cloud_downsampling(points, colors, k, batch_size=10000):
    """Perform strict 1/k downsampling with batch processing to reduce memory usage."""
    log(f"Performing strict 1/{k} downsampling with batch size {batch_size}")
    
    num_samples = len(points) // k  # 정확히 1/k로 샘플링
    sampled_indices = np.random.choice(len(points), num_samples, replace=False)
    sampled_points = points[sampled_indices]
    
    final_sampled_points = []
    final_sampled_colors = [] if colors is not None else None
    
    start_time = time.time()
    
    # ✅ KDTree를 한 번만 생성하여 메모리 절약
    tree = cKDTree(points)
    
    for i in range(0, len(sampled_points), batch_size):
        batch = sampled_points[i:i + batch_size]
        _, idx = tree.query(batch, k=k)
        
        sampled_batch = np.mean(points[idx], axis=1)
        final_sampled_points.extend(sampled_batch)
        
        if colors is not None:
            sampled_color_batch = np.mean(colors[idx] ** 2, axis=1) ** 0.5
            final_sampled_colors.extend(sampled_color_batch)
        
        log(f"Processed {i + len(batch)} points... (Elapsed: {time.time() - start_time:.2f}s)", overwrite=True)
    
    log("")
    return np.array(final_sampled_points), (np.array(final_sampled_colors) if colors is not None else None)

def process_input(input_path, output_dir, k):
    """Process a single file or all point cloud files in a directory."""
    if os.path.isfile(input_path):
        file_name = os.path.basename(input_path)
        log(f"Processing single file: {file_name}")
        points, colors = load_point_cloud(input_path)
        sampled_points, sampled_colors = sample_point_cloud_downsampling(points, colors, k)
        save_point_cloud(sampled_points, sampled_colors, output_dir, file_name, k)
        log(f"Completed processing: {file_name}")
    elif os.path.isdir(input_path):
        log(f"Processing directory: {input_path}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for file_name in os.listdir(input_path):
            if file_name.endswith('.ply') or file_name.endswith('.pcd'):
                log(f"Processing file: {file_name}")
                input_file = os.path.join(input_path, file_name)
                points, colors = load_point_cloud(input_file)
                sampled_points, sampled_colors = sample_point_cloud_downsampling(points, colors, k)
                save_point_cloud(sampled_points, sampled_colors, output_dir, file_name, k)
                log(f"Completed processing: {file_name}")
    else:
        log(f"Error: {input_path} is not a valid file or directory.")

def main():
    """Parse command line arguments and initiate processing."""
    parser = argparse.ArgumentParser(description='Strict 1/k Downsampling of Point Cloud')
    parser.add_argument('-ip', '--input_path', required=True, help='Input file or directory containing point cloud files')
    parser.add_argument('-op', '--output_path', required=True, help='Output directory for sampled point cloud files')
    parser.add_argument('-k', '--neighbors', type=int, required=True, help='Strict downsampling factor (1/k of points will be retained)')
    
    args = parser.parse_args()
    log(f"Starting strict 1/{args.neighbors} downsampling")
    process_input(args.input_path, args.output_path, args.neighbors)
    log("Processing complete.")

if __name__ == "__main__":
    main()
