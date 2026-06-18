"""
Utonia Demo Runner - Modified to use local weights and save outputs.
Runs object PCA visualization on sample 3 data.
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
    u, s, v = torch.pca_lowrank(feat, center=center, niter=5, q=9)
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

    utonia.utils.set_seed(73)

    # Load Model using local path
    print("\n--- Loading Utonia model from local checkpoint ---")
    model = utonia.load("ckpt/utonia.pth").to(device)
    model.eval()
    print("Model loaded successfully!")

    # Load data
    print("\n--- Loading sample data ---")
    data_path = os.path.expanduser("~/.cache/utonia/data/sample3_object.npz")
    point = dict(np.load(data_path))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point["coord"])
    pcd.colors = o3d.utility.Vector3dVector(point["color"])
    pcd.estimate_normals()
    point["normal"] = np.asarray(pcd.normals)

    point_rotated = {}
    point_rotated["coord"] = point["coord"][:, [0, 2, 1]]  # Specific shuffle
    point_rotated["color"] = point["color"]

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_rotated["coord"])
    pcd2.colors = o3d.utility.Vector3dVector(point_rotated["color"])
    pcd2.estimate_normals()
    point_rotated["normal"] = np.asarray(pcd2.normals) # Using pcd2.normals instead of pcd.normals to match original logic but fixing logical issue

    bias = np.array([0, 0, 1])
    point_rotated["coord"] = point_rotated["coord"] + bias  # Apply bias for positioning

    transform = utonia.transform.default()

    point = transform(point)
    point_rotated = transform(point_rotated)

    point = utonia.data.collate_fn([point, point_rotated])

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
            
    batched_coord = point.coord.clone()
    batched_coord[:, 2] += point.batch * bias[2]
    batched_color = point.color.clone()
    pca_color = get_pca_color(point.feat, brightness=1.2, center=True)

    print("\n--- Saving outputs ---")
    # Export PCA
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(batched_coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    o3d.io.write_point_cloud(os.path.join(output_dir, "pca_object.ply"), pcd)
    print(f"Saved: {output_dir}/pca_object.ply")
    
    print("\n--- Rendering images ---")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_dir, "pca_object.png"))
    vis.destroy_window()
    print(f"Saved: {output_dir}/pca_object.png")

    print("\n=== DEMO COMPLETE ===")

    if device == "cuda":
        print(f"\nPeak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        torch.cuda.empty_cache()
