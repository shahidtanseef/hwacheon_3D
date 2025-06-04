import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

# Configure and start the pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting RealSense pipeline...")
pipeline.start(config)
print("Pipeline started.")

# Align depth to color
align = rs.align(rs.stream.color)
pc = rs.pointcloud()

# Setup Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("RealSense Point Cloud", width=960, height=540)
pointcloud = o3d.geometry.PointCloud()
vis.add_geometry(pointcloud)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Waiting for valid frames...")
            continue

        # Generate point cloud
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        color_image = np.asanyarray(color_frame.get_data())

        # Filter out zero points
        valid = np.logical_and(np.abs(vtx[:, 2]) > 0.1, np.abs(vtx[:, 2]) < 2.5)
        vtx = vtx[valid]
        tex = tex[valid]

        h, w, _ = color_image.shape
        colors = []
        for u, v in tex:
            x = int(min(max(u * w, 0), w - 1))
            y = int(min(max(v * h, 0), h - 1))
            colors.append(color_image[y, x] / 255.0)

        # Update Open3D point cloud
        pointcloud.points = o3d.utility.Vector3dVector(vtx)
        pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    pipeline.stop()
    vis.destroy_window()
    print("Pipeline stopped and window closed.")
