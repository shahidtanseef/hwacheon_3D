import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)
pc = rs.pointcloud()

# Warm-up camera
for _ in range(30):
    pipeline.wait_for_frames()

# Capture one frame
frames = pipeline.wait_for_frames()
aligned = align.process(frames)
depth = aligned.get_depth_frame()
color = aligned.get_color_frame()

# Map point cloud to color
pc.map_to(color)
points = pc.calculate(depth)

# Convert to numpy
vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
color_image = np.asanyarray(color.get_data())
h, w, _ = color_image.shape

# Filter valid depth
valid = np.logical_and(np.abs(vtx[:, 2]) > 0.1, np.abs(vtx[:, 2]) < 2.5)
vtx = vtx[valid]
tex = tex[valid]

# Color mapping
colors = []
for u, v in tex:
    x = int(min(max(u * w, 0), w - 1))
    y = int(min(max(v * h, 0), h - 1))
    colors.append(color_image[y, x] / 255.0)

# Create and save point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vtx)
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

o3d.io.write_point_cloud("realsense_snapshot.ply", pcd)
print("Saved point cloud to realsense_snapshot.ply")

pipeline.stop()
