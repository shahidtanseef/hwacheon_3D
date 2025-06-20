import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import time


time.sleep(5)
# Create output directory
os.makedirs("recorded_depth", exist_ok=True)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

# Warm-up
for _ in range(30):
    pipeline.wait_for_frames()

# Set up video writer for RGB
video_writer = cv2.VideoWriter("rgb.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

print("Recording for 3 seconds...")
start_time = time.time()
frame_id = 0

while time.time() - start_time < 3:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth = aligned.get_depth_frame()
    color = aligned.get_color_frame()

    if not depth or not color:
        continue

    # Save color frame to video
    color_image = np.asanyarray(color.get_data())
    video_writer.write(color_image)

    # Save depth frame as raw array
    depth_image = np.asanyarray(depth.get_data())
    np.save(f"recorded_depth/depth_{frame_id:04d}.npy", depth_image)

    frame_id += 1

# Clean up
pipeline.stop()
video_writer.release()
print(f"Saved {frame_id} RGB frames to 'rgb.avi' and depth frames to 'recorded_depth/'")
