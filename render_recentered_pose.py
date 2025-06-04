import open3d as o3d
import numpy as np
import cv2
import os
import json
from math import sin, cos, radians

# ---------- CONFIG ----------
model_path = "manijaCavity/mesh/MANIJA_CAVITY_ROTATE.obj"
intr_path = "intrinsics.json"
output_img = "render_centered.png"
align_principal_axes = True  # Set to False to only center
# -----------------------------

# Load intrinsics
with open(intr_path, "r") as f:
    intr = json.load(f)
image_width = intr["width"]
image_height = intr["height"]

# Load mesh
mesh = o3d.io.read_triangle_mesh(model_path)
mesh.compute_vertex_normals()

# Center it
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
mesh.translate(-center)

# Align with principal axes if requested
if align_principal_axes:
    print("\nAligning with principal axes...")
    coords = np.asarray(mesh.vertices)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    mesh.rotate(eigvecs, center=(0, 0, 0))

# Print new bounding box
bbox_new = mesh.get_axis_aligned_bounding_box()
print("\n✅ Transformed Object Info:")
print(f"- New center: {bbox_new.get_center()}")
print(f"- New bounding box min: {bbox_new.min_bound}")
print(f"- New bounding box max: {bbox_new.max_bound}")

# Add XYZ axes at origin
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# Setup renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(image_width, image_height)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
scene = renderer.scene
scene.set_background([1, 1, 1, 1])
scene.add_geometry("mesh", mesh, mat)
scene.add_geometry("axis", axis, o3d.visualization.rendering.MaterialRecord())

# Camera location (front-right-up view)
radius = np.linalg.norm(bbox_new.get_extent()) * 2.5
cam_pos = np.array([radius, radius, radius])
scene.camera.look_at([0, 0, 0], cam_pos, [0, 0, 1])

# Render and save
img = renderer.render_to_image()
img_np = np.asarray(img)
cv2.imwrite(output_img, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
print(f"\n🖼️ Saved centered & aligned render to: {output_img}")
