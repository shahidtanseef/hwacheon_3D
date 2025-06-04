import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
CUP_CLASS_ID = 41

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Camera intrinsics
depth_sensor = profile.get_device().first_depth_sensor()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Pixel to real-world conversion
def pixel_to_metric(w_px, h_px, depth_m):
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy
    w_m = (w_px * depth_m) / fx
    h_m = (h_px * depth_m) / fy
    return w_m, h_m

# Draw XYZ arrows at 3D point
def draw_xyz_arrows(img, origin_3d, intrinsics, length=0.07):
    x_end = [origin_3d[0] + length, origin_3d[1], origin_3d[2]]
    y_end = [origin_3d[0], origin_3d[1] + length, origin_3d[2]]
    z_end = [origin_3d[0], origin_3d[1], origin_3d[2] + length]

    origin_2d = rs.rs2_project_point_to_pixel(intrinsics, origin_3d)
    x_2d = rs.rs2_project_point_to_pixel(intrinsics, x_end)
    y_2d = rs.rs2_project_point_to_pixel(intrinsics, y_end)
    z_2d = rs.rs2_project_point_to_pixel(intrinsics, z_end)

    origin = tuple(map(int, origin_2d))
    x_pt = tuple(map(int, x_2d))
    y_pt = tuple(map(int, y_2d))
    z_pt = tuple(map(int, z_2d))

    cv2.arrowedLine(img, origin, x_pt, (0, 0, 255), 2, tipLength=0.2)  # X - red
    cv2.arrowedLine(img, origin, y_pt, (0, 255, 0), 2, tipLength=0.2)  # Y - green
    cv2.arrowedLine(img, origin, z_pt, (255, 0, 0), 2, tipLength=0.2)  # Z - blue

# Main loop
cv2.namedWindow("Cup Detection with Depth")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model(color_image, verbose=False)[0]

        for box in results.boxes:
            if int(box.cls[0]) != CUP_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            width_px, height_px = x2 - x1, y2 - y1

            # Depth at center
            depth_center = depth_frame.get_distance(cx, cy)

            # Patch above cup for thickness
            patch_depths = []
            y_patch = max(y1 - 10, 0)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px = np.clip(cx + dx, 0, 639)
                    py = np.clip(y_patch + dy, 0, 479)
                    d = depth_frame.get_distance(px, py)
                    if d > 0:
                        patch_depths.append(d)

            thickness_m = max(0, np.mean(patch_depths) - depth_center) if patch_depths else 0
            width_m, height_m = pixel_to_metric(width_px, height_px, depth_center)

            # 3D position
            cup_coords = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth_center)

            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Info labels
            info_lines = [
                f"Distance: {depth_center:.2f} m",
                f"Height:   {height_m*100:.1f} cm",
                f"Width:    {width_m*100:.1f} cm",
                f"Height (px): {height_px}",
                f"Width (px):  {width_px}",
                f"Depth: {thickness_m*100:.1f} cm"
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(color_image, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw XYZ coordinates label above arrows
            x3d, y3d, z3d = cup_coords
            coord_text = f"X:{x3d:.3f} Y:{y3d:.3f} Z:{z3d:.3f}"
            label_pos = (cx - 50, cy - 30)
            cv2.putText(color_image, coord_text, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # yellow
            cv2.circle(color_image, (cx, cy), 5, (255, 0, 255), -1)

            draw_xyz_arrows(color_image, cup_coords, color_intrinsics)
            break  # Only first cup

        # Combine views
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        stacked = np.hstack((color_image, depth_colormap))
        cv2.imshow("Cup Detection with Depth", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
