import pyrealsense2 as rs
import torch
import numpy as np
import cv2
import time
from segment_anything import sam_model_registry, SamPredictor

# ========== Load SAM model ==========
checkpoint_path = "sam_vit_l_0b3195.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# ========== Start RealSense ==========
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# ========== Helper Functions ==========
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

# ========== Warm-up Camera ==========
print("Warming up RealSense (skipping 50 frames)...")
for _ in range(50):
    pipeline.wait_for_frames()
print("Done warming up.")

# ========== Capture Aligned Frame ==========
frames = pipeline.wait_for_frames()
aligned = align.process(frames)
depth_frame = aligned.get_depth_frame()
color_frame = aligned.get_color_frame()
pipeline.stop()

depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())
image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# ========== Define Prompt from Closest Center Pixel ==========
h, w = depth_image.shape
cx, cy = w // 2, h // 2
crop = depth_image[cy - 20:cy + 20, cx - 20:cx + 20]
min_idx = np.unravel_index(np.argmin(crop), crop.shape)
min_y, min_x = min_idx
positive_point = np.array([cx - 20 + min_x, cy - 20 + min_y])

offset = 100
neg_points = [
    [positive_point[0] - offset, positive_point[1]],
    [positive_point[0] + offset, positive_point[1]]
]
neg_points = [p for p in neg_points if 0 <= p[0] < w and 0 <= p[1] < h and depth_image[p[1], p[0]] > 0]
input_points = np.array([positive_point] + neg_points)
input_labels = np.array([1] + [0] * len(neg_points))

# ========== Run SAM ==========
start_time = time.time()
masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)
inference_time = time.time() - start_time
mask = masks[np.argmax(scores)]

# ========== Extract Mask Area ==========
ys, xs = np.where(mask)
x0, y0, x1, y1 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
bbox_center = ((x0 + x1) // 2, (y0 + y1) // 2)

# ========== Sample Depth Only Inside Mask Center ==========
patch_radius = 5
px, py = bbox_center
patch_depths = []
for dx in range(-patch_radius, patch_radius + 1):
    for dy in range(-patch_radius, patch_radius + 1):
        x = np.clip(px + dx, 0, w - 1)
        y = np.clip(py + dy, 0, h - 1)
        if mask[y, x]:
            d = depth_image[y, x]
            if d > 0:
                patch_depths.append(d)

if patch_depths:
    center_depth = np.median(patch_depths)
else:
    center_depth = 0

# ========== Convert to Real World Metrics ==========
if center_depth > 0:
    X, Y, Z = pixel_to_xyz(bbox_center[0], bbox_center[1], center_depth)
    width_m, height_m = pixel_to_metric(x1 - x0, y1 - y0, center_depth)
else:
    X = Y = Z = width_m = height_m = 0

# ========== Background Patch for Thickness ==========
bg_patch = []
for dx in range(-10, 11):
    for dy in range(-10, 11):
        px = np.clip(x0 + dx, 0, w - 1)
        py = np.clip(y0 - 15 + dy, 0, h - 1)
        d = depth_image[py, px]
        if d > 0:
            bg_patch.append(d)
if bg_patch and center_depth > 0:
    bg_depth = np.mean(bg_patch)
    thickness_m = max(0, bg_depth - center_depth)
else:
    thickness_m = 0

# ========== Visualization ==========
overlay = color_image.copy()
overlay[mask] = [0, 255, 0]
blended = cv2.addWeighted(color_image, 0.5, overlay, 0.5, 0)
cv2.rectangle(blended, (x0, y0), (x1, y1), (0, 255, 255), 2)
cv2.circle(blended, tuple(bbox_center), 5, (0, 255, 0), -1)

info = [
    f"X: {X:.3f} m, Y: {Y:.3f} m, Z: {Z:.3f} m",
    f"Width: {width_m*100:.1f} cm, Height: {height_m*100:.1f} cm",
    f"Thickness (depth diff): {thickness_m*100:.1f} cm",
    f"Inference: {inference_time:.2f} s"
]

for i, line in enumerate(info):
    cv2.putText(blended, line, (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ========== Show Output ==========
cv2.imshow("SAM with Real Depth (Final)", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
