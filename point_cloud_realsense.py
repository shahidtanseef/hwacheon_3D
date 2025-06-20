import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time

print("Initializing RealSense pipeline...")

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)
pc = rs.pointcloud()

# Warm-up
print("Warming up camera...")
for _ in range(30):
    try:
        pipeline.wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        pass

print("Capturing frames for 5 seconds...")
frames_list = []
start_time = time.time()

# === Stage 1: FAST frame capture loop ===
while time.time() - start_time < 5:
    try:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        aligned = align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if depth and color:
            frames_list.append((depth, color))
            print(f"Captured frame {len(frames_list)}")
    except RuntimeError:
        print("Warning: Timeout, skipping frame.")
        continue

pipeline.stop()
print(f"Captured {len(frames_list)} frames. Now processing...")

# === Stage 2: Point cloud generation ===
all_points = []
all_colors = []

for idx, (depth, color) in enumerate(frames_list):
    pc.map_to(color)
    points = pc.calculate(depth)

    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color.get_data())
    h, w, _ = color_image.shape

    valid = np.logical_and(vtx[:, 2] > 0.1, vtx[:, 2] < 2.5)
    vtx = vtx[valid]
    tex = tex[valid]

    colors = []
    for u, v in tex:
        x = int(min(max(u * w, 0), w - 1))
        y = int(min(max(v * h, 0), h - 1))
        colors.append(color_image[y, x] / 255.0)

    all_points.append(vtx)
    all_colors.append(np.array(colors))
    print(f"Processed frame {idx+1}/{len(frames_list)}")

# Combine and clean point cloud
print("Combining and cleaning point cloud...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

# Flip for Open3D visualization
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Downsample and remove outliers
pcd = pcd.voxel_down_sample(voxel_size=0.01)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Save and visualize
output_file = "final_scene_cloud_cleaned.ply"
o3d.io.write_point_cloud(output_file, pcd)
o3d.visualization.draw_geometries([pcd])
print(f"Saved cleaned point cloud to '{output_file}'")
