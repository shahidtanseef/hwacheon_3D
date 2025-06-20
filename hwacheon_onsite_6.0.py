import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# ------------------ Utility Functions ------------------

def get_valid_frames(pipeline, align, timeout=1000, retries=10):
    for _ in range(retries):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=timeout)
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if depth and color:
                return depth, color
        except RuntimeError:
            print("Retrying frame capture...")
    raise RuntimeError("Failed to get valid frames after multiple retries.")

def classify_mask_shape(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    c = max(contours, key=cv2.contourArea)
    area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
    if perimeter == 0:
        return "unknown"
    circ = 4 * np.pi * area / (perimeter ** 2)
    x, y, w, h = cv2.boundingRect(c)
    ar = w / h
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) >= 8 and 0.7 < circ < 1.2:
        return "circle"
    elif len(approx) == 4:
        return "square" if 0.8 < ar < 1.2 else "rectangle"
    return "irregular"

def get_bounding_box_and_shape(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, "unknown"
    c = max(contours, key=cv2.contourArea)
    x0, y0, w, h = cv2.boundingRect(c)
    bbox_center = (x0 + w // 2, y0 + h // 2)
    shape = classify_mask_shape(mask)
    return (x0, y0, w, h, bbox_center), c, shape

def pixel_to_xyz(x, y, depth_m, intr):
    cx, cy = intr.ppx, intr.ppy
    fx, fy = intr.fx, intr.fy
    X = (x - cx) * depth_m / fx
    Y = (y - cy) * depth_m / fy
    return X, Y, depth_m

def pixel_to_metric(w_px, h_px, depth_m, intr):
    return (w_px * depth_m) / intr.fx, (h_px * depth_m) / intr.fy

# ------------------ Setup ------------------

sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale:.6f} meters/unit")

depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Manual scaling correction factors (calibrate once with real object)
# Example: if measured was 10.5 cm, real was 6 cm ⇒ 6/10.5 ≈ 0.57
scale_factor_width = 0.57
scale_factor_height = 0.57

# ------------------ Capture Initial Frame ------------------

for _ in range(30):
    pipeline.wait_for_frames()

depth_init, color_init = get_valid_frames(pipeline, align)
depth_img_init = np.asanyarray(depth_init.get_data()) * depth_scale
color_img_init = np.asanyarray(color_init.get_data())
image_rgb = cv2.cvtColor(color_img_init, cv2.COLOR_BGR2RGB)

input_points = []
input_labels = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(1)
        print(f"Positive: {x}, {y}")
        cv2.circle(color_img_init, (x, y), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(0)
        print(f"Negative: {x}, {y}")
        cv2.circle(color_img_init, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow("Select object", color_img_init)

cv2.imshow("Select object", color_img_init)
cv2.setMouseCallback("Select object", click_event)
print("Left-click = positive, Right-click = negative. Press any key when done.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if not input_points:
    raise ValueError("No points selected.")

input_points = np.array(input_points)
input_labels = np.array(input_labels)

# ------------------ Segment ------------------

predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
mask = masks[np.argmax(scores)]
mask_resized = cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)

# ------------------ Bounding Box & Shape ------------------

bbox_info, contour, shape_type = get_bounding_box_and_shape(mask_resized)
if not bbox_info:
    raise RuntimeError("No valid contour found in mask.")

x0, y0, w, h, bbox_center = bbox_info
print(f"Shape detected: {shape_type}, BBox: {x0}, {y0}, {w}, {h}")

# ------------------ Live Tracking ------------------

print("Tracking started. Press 'q' to quit.")
saved = False

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image / depth_scale, alpha=0.03), cv2.COLORMAP_JET)

        patch_depths = [
            depth_image[np.clip(bbox_center[1] + dy, 0, 479), np.clip(bbox_center[0] + dx, 0, 639)]
            for dx in range(-5, 6) for dy in range(-5, 6)
            if mask_resized[np.clip(bbox_center[1] + dy, 0, 479), np.clip(bbox_center[0] + dx, 0, 639)] > 0
        ]
        patch_depths = [d for d in patch_depths if 0.1 < d < 2.0]

        if patch_depths:
            center_depth = np.median(patch_depths)
            X, Y, Z = pixel_to_xyz(*bbox_center, center_depth, depth_intr)
            width_m_raw, height_m_raw = pixel_to_metric(w, h, center_depth, depth_intr)
            width_m = width_m_raw * scale_factor_width
            height_m = height_m_raw * scale_factor_height
        else:
            X = Y = Z = width_m = height_m = center_depth = None

        bg_depths = [
            depth_image[y0 - 10, x]
            for x in range(x0, x0 + w)
            if 0.1 < depth_image[y0 - 10, x] < 2.0
        ]
        thickness_m = max(0, np.mean(bg_depths) - center_depth) if bg_depths and center_depth else None

        cv2.drawContours(color_image, [contour], -1, (0, 255, 255), 2)
        cv2.circle(color_image, bbox_center, 5, (0, 255, 0), -1)

        info = [
            f"X: {X:.3f} m, Y: {Y:.3f} m, Z: {Z:.3f} m" if X else "Position: N/A",
            f"Width: {width_m*100:.1f} cm, Height: {height_m*100:.1f} cm" if width_m else "Size: N/A",
            f"Depth: {thickness_m*100:.1f} cm" if thickness_m else "Depth: N/A",
            f"Shape: {shape_type}"
            #print
        ]
        #print("width:",width_m*100)
        for i, line in enumerate(info):
            cv2.putText(color_image, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        stacked = np.hstack((color_image, depth_colormap))
        cv2.imshow("SAM + RealSense | Depth", stacked)

        if center_depth and thickness_m and not saved:
            with open("object_info.txt", "w") as f:
                f.write(f"X: {X:.3f} m\n")
                f.write(f"Y: {Y:.3f} m\n")
                f.write(f"Z: {Z:.3f} m\n")
                f.write(f"Width: {width_m*100:.1f} cm\n")
                f.write(f"Height: {height_m*100:.1f} cm\n")
                f.write(f"Thickness: {thickness_m*100:.1f} cm\n")
                f.write(f"Shape: {shape_type}\n")
            saved = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
