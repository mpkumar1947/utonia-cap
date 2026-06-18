"""
Utonia Demo Runner - Modified to use local weights and save outputs.
Runs semantic segmentation visualization on sample indoor point cloud data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utonia
import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ScanNet Meta data
VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0), 1: (174.0, 199.0, 232.0), 2: (152.0, 223.0, 138.0), 3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0), 5: (188.0, 189.0, 34.0), 6: (140.0, 86.0, 75.0), 7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0), 9: (197.0, 176.0, 213.0), 10: (148.0, 103.0, 189.0), 11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0), 14: (247.0, 182.0, 210.0), 15: (66.0, 188.0, 102.0), 16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0), 18: (202.0, 185.0, 52.0), 19: (51.0, 176.0, 203.0), 20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0), 22: (78.0, 71.0, 183.0), 23: (172.0, 114.0, 82.0), 24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0), 26: (153.0, 98.0, 156.0), 27: (140.0, 153.0, 101.0), 28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0), 30: (178.0, 127.0, 135.0), 32: (146.0, 111.0, 194.0), 33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0), 35: (96.0, 207.0, 209.0), 36: (227.0, 119.0, 194.0), 37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0), 39: (82.0, 84.0, 163.0), 40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]

class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # set random seed
    utonia.utils.set_seed(46647087)

    # Load model
    print("\n--- Loading Utonia model from local checkpoint ---")
    model = utonia.load("ckpt/utonia.pth").to(device)
    model.eval()
    print(f"Model loaded successfully!")
    
    # Load linear probing seg head
    print("\n--- Loading segmentation head from local checkpoint ---")
    ckpt = torch.load("ckpt/utonia_linear_prob_head_sc.pth", map_location=device, weights_only=False)
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])
    seg_head.eval()
    print("Segmentation head loaded successfully!")

    # Load default data transform pipeline
    transform = utonia.transform.default(0.5)

    # Load data
    print("\n--- Loading sample data ---")
    data_path = os.path.expanduser("~/.cache/utonia/data/sample1.npz")
    point = dict(np.load(data_path))

    point.pop("segment200")
    segment = point.pop("segment20")
    point["segment"] = segment
    original_coord = point["coord"].copy()
    point = transform(point)

    # Inference
    print("\n--- Running inference ---")
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]

    # Map color back to original scale if needed
    # The output 'point.coord' represents the downsampled points if 'transform' uses GridSampling.
    # We will use the output point clouds directly.

    print("\n--- Saving outputs ---")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color / 255.0)
    o3d.io.write_point_cloud(os.path.join(output_dir, "sem_seg.ply"), pcd)
    print(f"Saved: {output_dir}/sem_seg.ply")

    # Render and save image
    print("\n--- Rendering images ---")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_dir, "sem_seg_indoor.png"))
    vis.destroy_window()
    print(f"Saved: {output_dir}/sem_seg_indoor.png")

    print("\n=== DEMO COMPLETE ===")

    if device == "cuda":
        print(f"\nPeak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        torch.cuda.empty_cache()
