import open3d as o3d
import numpy as np
import cv2
import os
import json
from math import sin, cos, radians

# ---------- CONFIG ----------
model_path = "manijaCavity/mesh/MANIJA_CAVITY_ROTATE.obj"
intr_path = "intrinsics.json"
output_img = "render_with_axes.png"
# -----------------------------

# Load intrinsics
with open(intr_path, "r") as f:
    intr = json.load(f)
image_width = intr["width"]
image_height = intr["height"]

# Load mesh
mesh = o3d.io.read_triangle_mesh(model_path)
mesh.compute_vertex_normals()

# Analyze pose and bounding box
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
radius = np.linalg.norm(extent) * 2.5

print("\n🧭 Default Object Pose Information:")
print(f"- Bounding box min: {bbox.min_bound}")
print(f"- Bounding box max: {bbox.max_bound}")
print(f"- Object center: {center}")
print(f"- Extents (size): {extent}")
print(f"- Diagonal length (radius): {radius:.4f} m")

# PCA for principal axes
coords = np.asarray(mesh.vertices) - center
cov = np.cov(coords.T)
eigvals, eigvecs = np.linalg.eigh(cov)
print("\n📐 Principal Axes (from PCA):")
for i in range(3):
    print(f"Axis {i+1}: direction = {eigvecs[:, i]}, variance = {eigvals[i]:.6f}")

# Add XYZ coordinate frame at the object center
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
axis.translate(center)

# Setup renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(image_width, image_height)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
scene = renderer.scene
scene.set_background([1, 1, 1, 1])
scene.add_geometry("object", mesh, mat)
scene.add_geometry("axis", axis, o3d.visualization.rendering.MaterialRecord())

# Choose one camera angle (e.g., 45° azimuth, 45° elevation)
elev_deg = 45
azim_deg = 45
elev_rad = radians(elev_deg)
azim_rad = radians(azim_deg)
cam_x = center[0] + radius * cos(elev_rad) * cos(azim_rad)
cam_y = center[1] + radius * cos(elev_rad) * sin(azim_rad)
cam_z = center[2] + radius * sin(elev_rad)
cam_pos = np.array([cam_x, cam_y, cam_z])
up = np.array([0, 0, 1])

# Set camera to look at object
scene.camera.look_at(center, cam_pos, up)

# Render and save image
img = renderer.render_to_image()
img_np = np.asarray(img)
cv2.imwrite(output_img, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
print(f"\n✅ Saved image with XYZ axes to '{output_img}'")
