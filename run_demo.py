"""
Utonia Demo Runner - Modified to use local weights and save outputs.
Runs PCA visualization on sample indoor point cloud data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utonia
import torch
import open3d as o3d
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    utonia.utils.set_seed(37)

    # Load model from local checkpoint
    print("\n--- Loading Utonia model from local checkpoint ---")
    model = utonia.load("ckpt/utonia.pth").to(device)
    model.eval()
    print(f"Model loaded successfully!")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    # Load default data transform pipeline
    transform = utonia.transform.default(0.5)

    # Load sample data from local cache
    print("\n--- Loading sample data ---")
    data_path = os.path.expanduser("~/.cache/utonia/data/sample1.npz")
    point = dict(np.load(data_path))
    print(f"Point cloud keys: {list(point.keys())}")
    print(f"Number of points: {point['coord'].shape[0]}")
    print(f"Coord shape: {point['coord'].shape}")
    print(f"Color shape: {point['color'].shape}")
    print(f"Normal shape: {point['normal'].shape}")

    point.pop("segment200")
    segment = point.pop("segment20")
    point["segment"] = segment
    original_coord = point["coord"].copy()
    original_color = point["color"].copy() / 255.0

    # Transform data
    print("\n--- Transforming data ---")
    point = transform(point)

    # Run inference
    print("\n--- Running inference ---")
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)

        # Check GPU memory before forward
        if device == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        # Model forward
        point = model(point)

        if device == "cuda":
            print(f"GPU memory after forward: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        # Upcast point features
        print("Upcasting features...")
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        # Feature info
        print(f"Feature shape (after upcast): {point.feat.shape}")
        feat_original = point.feat[point.inverse]
        print(f"Feature shape (original scale): {feat_original.shape}")

        # PCA visualization
        print("\n--- Computing PCA colors ---")
        pca_color = get_pca_color(point.feat, brightness=1.2, center=True)

    # Map back to original scale
    original_pca_color = pca_color[point.inverse]

    # Save original point cloud
    print("\n--- Saving outputs ---")
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_coord)
    pcd_original.colors = o3d.utility.Vector3dVector(original_color)
    o3d.io.write_point_cloud(os.path.join(output_dir, "original.ply"), pcd_original)
    print(f"Saved: {output_dir}/original.ply")

    # Save PCA-colored point cloud
    pcd_pca = o3d.geometry.PointCloud()
    pcd_pca.points = o3d.utility.Vector3dVector(original_coord)
    pcd_pca.colors = o3d.utility.Vector3dVector(original_pca_color.cpu().detach().numpy())
    o3d.io.write_point_cloud(os.path.join(output_dir, "pca_indoor.ply"), pcd_pca)
    print(f"Saved: {output_dir}/pca_indoor.ply")

    # Render and save images using offscreen rendering
    print("\n--- Rendering images ---")
    # Set up a visualizer for offscreen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)

    # Render original
    vis.add_geometry(pcd_original)
    vis.get_view_control().set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_dir, "original_indoor.png"))
    vis.clear_geometries()
    print(f"Saved: {output_dir}/original_indoor.png")

    # Render PCA
    vis.add_geometry(pcd_pca)
    vis.get_view_control().set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_dir, "pca_indoor.png"))
    vis.destroy_window()
    print(f"Saved: {output_dir}/pca_indoor.png")

    print("\n=== DEMO COMPLETE ===")
    print(f"All outputs saved to: {output_dir}/")

    if device == "cuda":
        print(f"\nPeak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        torch.cuda.empty_cache()
