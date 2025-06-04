import open3d as o3d
import numpy as np
import cv2
import os
import json
from math import radians, sin, cos

# ---------- CONFIG ----------
model_path = "manijaCavity/mesh/MANIJA_CAVITY_ROTATE.obj"
intr_path = "intrinsics.json"
output_dir = "renders_yaxis"
os.makedirs(output_dir, exist_ok=True)
pose_file = os.path.join(output_dir, "poses.txt")

n_views = 12           # 360° / 30° steps
radius_scale = 1.0     # reduced from 2.0 so object is visible
# ----------------------------

# Load RealSense intrinsics
with open(intr_path, "r") as f:
    intr = json.load(f)
W, H = intr["width"], intr["height"]
K = np.array(intr["K"]).astype(np.float64)

# Load mesh
mesh = o3d.io.read_triangle_mesh(model_path)
mesh.compute_vertex_normals()

# Fix 1: Center mesh at origin
mesh_center = mesh.get_center()
mesh.translate(-mesh_center)

# Fix 2: Rescale mesh if very small (assumes unit = meters)
bbox = mesh.get_axis_aligned_bounding_box()
extent = bbox.get_extent()
if np.linalg.norm(extent) < 0.1:  # likely in mm
    print("Rescaling mesh by 1000 (mm to meters)")
    mesh.scale(1000.0, center=(0, 0, 0))

# New bounds after fix
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = np.linalg.norm(bbox.get_extent()) * radius_scale

# Setup renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
scene = renderer.scene
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
scene.set_background([1, 1, 1, 1])
scene.add_geometry("object", mesh, mat)

# Look-at matrix builder
def look_at(cam_pos, target, up=np.array([0, 0, 1])):
    z = cam_pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack((x, y, z), axis=1)
    T = -R.T @ cam_pos
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = T
    return extrinsic

# Render loop
view_idx = 0
with open(pose_file, "w") as f:
    for i in range(n_views):
        angle_deg = (360 / n_views) * i
        angle_rad = radians(angle_deg)

        # Camera on horizontal ring around object
        x = center[0] + radius * cos(angle_rad)
        y = center[1] + radius * sin(angle_rad)
        z = center[2] + radius * 0.1
        cam_pos = np.array([x, y, z])

        extrinsic = look_at(cam_pos, center)

        # Apply camera settings
        scene.camera.set_projection(K, 0.01, 10.0, W, H)
        scene.camera.look_at(center, cam_pos, [0, 0, 1])

        # Render and save
        img = renderer.render_to_image()
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        img_path = os.path.join(output_dir, f"render_{view_idx:03d}.png")
        cv2.imwrite(img_path, gray)
        print(f"Saved {img_path}")

        # Save pose
        f.write(f"# View {view_idx}, azim {angle_deg:.1f}\n")
        np.savetxt(f, extrinsic, fmt="%.6f")
        f.write("\n")

        view_idx += 1

print(f"\n✅ Saved {view_idx} grayscale images and poses to '{output_dir}'")
