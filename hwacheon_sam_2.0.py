import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# ---------- Load SAM ----------
checkpoint_path = "sam_vit_l_0b3195.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# ---------- Start RealSense ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

def pixel_to_xyz(x, y, depth_m):
    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    X = (x - cx) * depth_m / fx
    Y = (y - cy) * depth_m / fy
    return X, Y, depth_m

def pixel_to_metric(w_px, h_px, depth_m):
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    w_m = (w_px * depth_m) / fx
    h_m = (h_px * depth_m) / fy
    return w_m, h_m

# ---------- Warm up ----------
print("Warming up RealSense...")
for _ in range(50):
    pipeline.wait_for_frames()
print("Capturing initial frame for SAM...")

# ---------- Capture initial frame ----------
frames = pipeline.wait_for_frames()
aligned = align.process(frames)
depth_frame_init = aligned.get_depth_frame()
color_frame_init = aligned.get_color_frame()
depth_image_init = np.asanyarray(depth_frame_init.get_data())
color_image_init = np.asanyarray(color_frame_init.get_data())
image_rgb = cv2.cvtColor(color_image_init, cv2.COLOR_BGR2RGB)
h, w = depth_image_init.shape

# ---------- Run SAM once ----------
predictor.set_image(image_rgb.copy())
cx, cy = w // 2, h // 2
crop = depth_image_init[cy - 20:cy + 20, cx - 20:cx + 20]
min_idx = np.unravel_index(np.argmin(crop), crop.shape)
min_y, min_x = min_idx
positive_point = np.array([cx - 20 + min_x, cy - 20 + min_y])
neg_points = [[positive_point[0] - 100, positive_point[1]], [positive_point[0] + 100, positive_point[1]]]
input_points = np.array([positive_point] + neg_points)
input_labels = np.array([1] + [0] * len(neg_points))

masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)
mask = masks[np.argmax(scores)]
mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
ys, xs = np.where(mask_resized)
x0, y0, x1, y1 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
bbox_center = ((x0 + x1) // 2, (y0 + y1) // 2)
print(bbox_center,"\n\n",x0, y0, x1, y1)

# ---------- Live tracking ----------
print("Tracking SAM box... Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- Depth at object center using get_distance ---
        cx, cy = bbox_center
        patch_depths = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x = np.clip(cx + dx, 0, w - 1)
                y = np.clip(cy + dy, 0, h - 1)
                if mask_resized[y, x]:
                    d = depth_frame.get_distance(x, y)
                    if 0.1 < d < 2.0:
                        patch_depths.append(d)

        valid_depth = len(patch_depths) > 0
        if valid_depth:
            center_depth = np.median(patch_depths)
            X, Y, Z = pixel_to_xyz(cx, cy, center_depth)
            width_m, height_m = pixel_to_metric(x1 - x0, y1 - y0, center_depth)
        else:
            center_depth = None
            X = Y = Z = width_m = height_m = None

        # --- Background thickness using get_distance ---
        bg_patch = []
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                px = np.clip(x0 + dx, 0, w - 1)
                py = np.clip(y0 - 15 + dy, 0, h - 1)
                d = depth_frame.get_distance(px, py)
                if 0.1 < d < 2.0:
                    bg_patch.append(d)

        bg_depth = np.mean(bg_patch) if bg_patch else None
        thickness_m = max(0, bg_depth - center_depth) if (bg_depth and valid_depth) else None

        # --- Visualization ---
        cv2.rectangle(color_image, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.circle(color_image, bbox_center, 5, (0, 255, 0), -1)

        info = []
        if valid_depth:
            info.append(f"X: {X:.3f} m, Y: {Y:.3f} m, Z: {Z:.3f} m")
            info.append(f"Width: {width_m*100:.1f} cm, Height: {height_m*100:.1f} cm")
        else:
            info.append("Position: N/A")
            info.append("Size: N/A")

        info.append(f"Depth: {thickness_m*100:.1f} cm" if thickness_m else "Depth: N/A")

        for i, line in enumerate(info):
            cv2.putText(color_image, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        stacked = np.hstack((color_image, depth_colormap))
        cv2.imshow("SAM + RealSense (Valid Depth Fixed)", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stopped.")
