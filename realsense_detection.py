import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8n model (replace with custom if needed)
model = YOLO("yolov8n.pt")
CUP_CLASS_ID = 41  # COCO class ID for "cup"

# Start RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Get intrinsics
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_stream = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

print("RealSense camera started. Press 'q' to quit.")

def pixel_to_metric(w_px, h_px, depth_m):
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy
    w_m = (w_px * depth_m) / fx
    h_m = (h_px * depth_m) / fy
    return w_m, h_m

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == CUP_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                width_px = x2 - x1
                height_px = y2 - y1

                # Depth at center of the cup
                depth_center = depth_frame.get_distance(cx, cy)

                # Patch above the bounding box (10 pixels up)
                patch_depths = []
                patch_half_size = 2  # 5x5 patch
                y_patch = max(y1 - 10, 0)

                for dx in range(-patch_half_size, patch_half_size + 1):
                    for dy in range(-patch_half_size, patch_half_size + 1):
                        px = np.clip(cx + dx, 0, 639)
                        py = np.clip(y_patch + dy, 0, 479)
                        d = depth_frame.get_distance(px, py)
                        if d > 0:
                            patch_depths.append(d)

                # Compute thickness from patch average
                if patch_depths and depth_center > 0:
                    depth_back = np.mean(patch_depths)
                    thickness_m = max(0, depth_back - depth_center)
                else:
                    thickness_m = 0

                # Real-world size
                width_m, height_m = pixel_to_metric(width_px, height_px, depth_center)

                # Prepare display text
                info_lines = [
                    f"Distance: {depth_center:.2f} m",
                    f"Height:   {height_m*100:.1f} cm",
                    f"Width:    {width_m*100:.1f} cm",
                    f"Height (px): {height_px}",
                    f"Width (px):  {width_px}",
                    f"Thickness: {thickness_m*100:.1f} cm"
                ]

                # Draw bounding box
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw text at top-left
                start_x, start_y = 10, 30
                line_height = 25
                for i, line in enumerate(info_lines):
                    y = start_y + i * line_height
                    cv2.putText(color_image, line, (start_x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show output
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        stacked = np.hstack((color_image, depth_colormap))
        cv2.imshow('Cup Detection with Full Metrics', stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("RealSense camera stopped.")
