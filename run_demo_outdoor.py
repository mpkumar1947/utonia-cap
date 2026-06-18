"""
Utonia Demo Runner - Modified to use local weights and save outputs.
Runs outdoor PCA visualization on sample 2 (multi-frame LiDAR) data.
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
    u, s, v = torch.pca_lowrank(feat, center=center, q=12, niter=5)
    projection = feat @ v
    projection = (
        projection[:, :3] * 0.4 + projection[:, 3:6] * 0.2 + projection[:, 9:12] * 0.4
    )
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    utonia.utils.set_seed(6985480)

    # Load Model using local path
    print("\n--- Loading Utonia model from local checkpoint ---")
    model = utonia.load("ckpt/utonia.pth").to(device)
    model.eval()
    print("Model loaded successfully!")

    # Load default data transform pipeline
    transform = utonia.transform.default(0.2, apply_z_positive=False)

    # Load data
    print("\n--- Loading sample outdoor data ---")
    data_path = os.path.expanduser("~/.cache/utonia/data/sample2_outdoor_multiframe.npz")
    point = dict(np.load(data_path))

    original_coord = point["coord"].copy()
    point = transform(point)

    print("\n--- Running inference ---")
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        # upcast point feature
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

        # PCA
        pca_color = get_pca_color(point.feat, brightness=1, center=True)

    original_pca_color = pca_color[point.inverse]

    print("\n--- Saving outputs ---")
    # Export PCA
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord)
    pcd.colors = o3d.utility.Vector3dVector(original_pca_color.cpu().detach().numpy())
    o3d.io.write_point_cloud(os.path.join(output_dir, "pca_outdoor.ply"), pcd)
    print(f"Saved: {output_dir}/pca_outdoor.ply")

    print("\n--- Rendering images ---")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)
    # Configure view for outdoor scene
    vc = vis.get_view_control()
    vc.set_zoom(0.2)
    vc.set_front([0.5, 0.5, -0.8])
    vc.set_lookat([0.0, 0.0, 0.0])
    vc.set_up([0.0, 0.0, 1.0])
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_dir, "pca_outdoor.png"))
    vis.destroy_window()
    print(f"Saved: {output_dir}/pca_outdoor.png")

    print("\n=== DEMO COMPLETE ===")

    if device == "cuda":
        print(f"\nPeak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        torch.cuda.empty_cache()
