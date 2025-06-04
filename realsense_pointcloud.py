import pyrealsense2 as rs
import open3d as o3d
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)
pc = rs.pointcloud()

for _ in range(30):
    frames = pipeline.wait_for_frames()

frames = align.process(pipeline.wait_for_frames())
depth = frames.get_depth_frame()
color = frames.get_color_frame()

# Generate RealSense-native point cloud
pc.map_to(color)
points = pc.calculate(depth)

# Convert to numpy
vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
color_image = np.asanyarray(color.get_data())
h, w, _ = color_image.shape

# Map texture manually
colors = []
for u, v in tex:
    x = min(w - 1, max(0, int(u * w)))
    y = min(h - 1, max(0, int(v * h)))
    colors.append(color_image[y, x] / 255.0)

# Construct Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vtx)
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# Flip for Open3D
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Save and visualize
o3d.io.write_point_cloud("d435i_cloud.ply", pcd)
o3d.visualization.draw_geometries([pcd])

pipeline.stop()
