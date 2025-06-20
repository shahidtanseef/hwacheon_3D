import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import torch
import time
from segment_anything import sam_model_registry, SamPredictor

# ------------------ Stage 1: Point Cloud Capture ------------------

print("Initializing RealSense pipeline...")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
pc = rs.pointcloud()

print("Warming up camera...")
for _ in range(30):
    try:
        pipeline.wait_for_frames(timeout_ms=1000)
    except RuntimeError:
        pass

print("Capturing frames for 2 seconds...")
frames_list = []
start_time = time.time()

while time.time() - start_time < 2:
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

pipeline.stop()
print("Captured {} frames. Now processing...".format(len(frames_list)))

all_points = []
all_colors = []

for idx, (depth, color) in enumerate(frames_list):
    pc.map_to(color)
    points = pc.calculate(depth)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color.get_data())
    h, w, _ = color_image.shape

    valid = np.logical_and(vtx[:, 2] > 0.05, vtx[:, 2] < 3.0)
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

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd = pcd.voxel_down_sample(voxel_size=0.005)

output_file = "final_scene_cloud_cleaned.ply"
o3d.io.write_point_cloud(output_file, pcd)
print(f"Saved cleaned point cloud to '{output_file}'")
print("Opening point cloud viewer... Close it to continue.")
o3d.visualization.draw_geometries([pcd])

# ------------------ Stage 2: SAM + RealSense Tracking ------------------

print("Point cloud closed. Starting SAM tracking...")

def pixel_to_xyz(x, y, depth_m, intrinsics):
    cx, cy = intrinsics.ppx, intrinsics.ppy
    fx, fy = intrinsics.fx, intrinsics.fy
    X = (x - cx) * depth_m / fx
    Y = (y - cy) * depth_m / fy
    return X, Y, depth_m

def pixel_to_metric(w_px, h_px, depth_m, intrinsics):
    fx, fy = intrinsics.fx, intrinsics.fy
    w_m = (w_px * depth_m) / fx
    h_m = (h_px * depth_m) / fy
    return w_m, h_m

def classify_mask_shape(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or area == 0:
        return "unknown"
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) >= 8 and 0.7 < circularity < 1.2:
        return "circle"
    elif len(approx) == 4:
        return "square" if 0.8 < aspect_ratio < 1.2 else "rectangle"
    return "irregular"

# Load SAM
checkpoint_path = "sam_vit_l_0b3195.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# Restart RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Warm up
for _ in range(30):
    pipeline.wait_for_frames()

# Capture initial frame for SAM
frames = pipeline.wait_for_frames()
aligned = align.process(frames)
depth_frame_init = aligned.get_depth_frame()
color_frame_init = aligned.get_color_frame()
depth_image_init = np.asanyarray(depth_frame_init.get_data())
color_image_init = np.asanyarray(color_frame_init.get_data())
image_rgb = cv2.cvtColor(color_image_init, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb.copy())

# --- Automatically select positive points based on closest depth ---
cx, cy = 320, 240
crop = depth_image_init[cy - 40:cy + 40, cx - 40:cx + 40].copy()
crop_mask = (crop > 0.1) & (crop < 2.0)

if not np.any(crop_mask):
    raise ValueError("No valid depth values in center crop.")

valid_depths = crop[crop_mask]
flat_indices = np.argsort(valid_depths)[:2]
valid_yx = np.argwhere(crop_mask)
selected_points = valid_yx[flat_indices] + [cy - 40, cx - 40]

input_points = np.array([[x, y] for y, x in selected_points])
input_labels = np.array([1] * len(input_points))  # Positives

# Add two negative points
input_points = np.vstack((input_points, [[20, 20], [620, 460]]))
input_labels = np.concatenate((input_labels, [0, 0]))

# SAM prediction
masks, scores, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
mask = masks[np.argmax(scores)]
mask_resized = cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)

# Bounding box from contour
binary = (mask_resized > 0).astype(np.uint8) * 255
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No object detected in mask.")
    exit()

largest_contour = max(contours, key=cv2.contourArea)
x0, y0, w, h = cv2.boundingRect(largest_contour)
x1 = x0 + w
y1 = y0 + h
bbox_center = (x0 + w // 2, y0 + h // 2)
shape_type = classify_mask_shape(mask_resized)

# Tracking loop
saved = False
print("SAM tracking started. Press 'q' to quit.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        patch_depths = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x = np.clip(bbox_center[0] + dx, 0, 639)
                y = np.clip(bbox_center[1] + dy, 0, 479)
                if mask_resized[y, x]:
                    d = depth_frame.get_distance(x, y)
                    if 0.1 < d < 2.0:
                        patch_depths.append(d)

        if patch_depths:
            center_depth = np.median(patch_depths)
            X, Y, Z = pixel_to_xyz(bbox_center[0], bbox_center[1], center_depth, depth_intrinsics)
            width_m, height_m = pixel_to_metric(x1 - x0, y1 - y0, center_depth, depth_intrinsics)
        else:
            X = Y = Z = width_m = height_m = center_depth = None

        bg_depths = [depth_frame.get_distance(x, y0 - 10)
                     for x in range(x0, x1)
                     if 0 <= x < 640 and 0.1 < depth_frame.get_distance(x, y0 - 10) < 2.0]
        thickness_m = max(0, np.mean(bg_depths) - center_depth) if bg_depths and center_depth else None

        cv2.rectangle(color_image, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.circle(color_image, bbox_center, 5, (0, 255, 0), -1)

        info = []
        if center_depth:
            info.append(f"X: {X:.3f} m, Y: {Y:.3f} m, Z: {Z:.3f} m")
            info.append(f"Width: {width_m*100:.1f} cm, Height: {height_m*100:.1f} cm")
        else:
            info.append("Position: N/A")
            info.append("Size: N/A")

        info.append(f"Depth: {thickness_m*100:.1f} cm" if thickness_m else "Depth: N/A")
        info.append(f"Shape: {shape_type}")

        for i, line in enumerate(info):
            cv2.putText(color_image, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
            print("Saved object info to 'object_info.txt'")

        cv2.imshow("SAM + RealSense", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
