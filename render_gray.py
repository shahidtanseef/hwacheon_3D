import open3d as o3d
import numpy as np
import cv2
import os
import json
from math import sin, cos, radians

# ---------- CONFIG ----------
model_path = "manijaCavity/mesh/MANIJA_CAVITY_ROTATE.obj"
intr_path = "intrinsics.json"
output_dir = "renders_6dof"
os.makedirs(output_dir, exist_ok=True)
pose_file = os.path.join(output_dir, "poses.txt")

azimuth_steps = 12     # 360° / 30°
elevation_steps = 5    # 0° to 75° (inclusive)
radius_scale = 2.5
image_width = 640
image_height = 480
# -----------------------------

# Load intrinsics
with open(intr_path, "r") as f:
    intr = json.load(f)
image_width = intr["width"]
image_height = intr["height"]

# Load mesh
mesh = o3d.io.read_triangle_mesh(model_path)
mesh.compute_vertex_normals()
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
radius = np.linalg.norm(extent) * radius_scale

# Setup renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(image_width, image_height)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
scene = renderer.scene
scene.set_background([1, 1, 1, 1])
scene.add_geometry("object", mesh, mat)

# Spherical sampling loop
view_idx = 0
with open(pose_file, "w") as f:
    for elev_step in range(elevation_steps):
        elev_deg = 15 + (75 / (elevation_steps - 1)) * elev_step  # from 15° to 90°
        elev_rad = radians(elev_deg)

        for azim_step in range(azimuth_steps):
            azim_deg = (360 / azimuth_steps) * azim_step
            azim_rad = radians(azim_deg)

            # Spherical coordinates
            cam_x = center[0] + radius * cos(elev_rad) * cos(azim_rad)
            cam_y = center[1] + radius * cos(elev_rad) * sin(azim_rad)
            cam_z = center[2] + radius * sin(elev_rad)
            cam_pos = np.array([cam_x, cam_y, cam_z])
            up = np.array([0, 0, 1]) if elev_deg < 85 else np.array([0, -1, 0])

            # Setup camera
            scene.camera.look_at(center, cam_pos, up)

            # Render
            img = renderer.render_to_image()
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            img_path = os.path.join(output_dir, f"render_{view_idx:03d}.png")
            cv2.imwrite(img_path, gray)
            print(f"Saved {img_path}")

            # Save pose
            extr = scene.camera.get_model_matrix()
            f.write(f"# View {view_idx}, elev {elev_deg:.1f}, azim {azim_deg:.1f}\n")
            np.savetxt(f, extr, fmt="%.6f")
            f.write("\n")

            view_idx += 1

print(f"\n✅ Saved {view_idx} views in '{output_dir}'")
print(f"✅ Saved all camera poses to '{pose_file}'")
