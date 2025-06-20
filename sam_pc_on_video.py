import numpy as np
import cv2
import glob
import open3d as o3d
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- Load Data ---
video = cv2.VideoCapture("rgb.avi")
depth_files = sorted(glob.glob("recorded_depth/depth_*.npy"))

# --- Dummy intrinsics (RealSense 640x480) ---
class DummyIntrinsics:
    fx = 615.0
    fy = 615.0
    ppx = 320.0
    ppy = 240.0

intrinsics = DummyIntrinsics()

# --- Utility Functions ---
def pixel_to_xyz(x, y, depth_m, intr):
    X = (x - intr.ppx) * depth_m / intr.fx
    Y = (y - intr.ppy) * depth_m / intr.fy
    return X, Y, depth_m

def pixel_to_metric(w_px, h_px, depth_m, intr):
    return (w_px * depth_m) / intr.fx, (h_px * depth_m) / intr.fy

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

def create_point_cloud(color_img, depth_img, intr):
    h, w = depth_img.shape
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img * 0.001
    mask = z > 0.1
    x, y, z = x[mask], y[mask], z[mask]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    points = np.stack((X, Y, z), axis=-1)
    colors = color_img[y, x] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# --- Load SAM ---
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# --- First Frame Initialization ---
ret, color_init = video.read()
depth_init = np.load(depth_files[0])
image_rgb = cv2.cvtColor(color_init, cv2.COLOR_BGR2RGB)

# --- Manual Click: 2 Positive + 2 Negative Points ---
selected_points = []
labels = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and labels.count(1) < 2:
        selected_points.append([x, y])
        labels.append(1)
        print(f"Positive point: {x}, {y}")
        cv2.circle(color_init, (x, y), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN and labels.count(0) < 2:
        selected_points.append([x, y])
        labels.append(0)
        print(f"Negative point: {x}, {y}")
        cv2.circle(color_init, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow("Click 2 positive (LMB) + 2 negative (RMB)", color_init)

cv2.imshow("Click 2 positive (LMB) + 2 negative (RMB)", color_init)
cv2.setMouseCallback("Click 2 positive (LMB) + 2 negative (RMB)", click_event)
print("Select 2 positive (left-click) and 2 negative (right-click) points. Then press any key.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(selected_points) < 4:
    print("Insufficient points selected. Exiting.")
    exit()

input_points = np.array(selected_points)
input_labels = np.array(labels)

# --- Run SAM ---
predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
mask = masks[np.argmax(scores)].astype(np.uint8)
mask_resized = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)

ys, xs = np.where(mask_resized)
x0, y0, x1, y1 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
bbox_center = ((x0 + x1) // 2, (y0 + y1) // 2)
shape_type = classify_mask_shape(mask_resized)

# --- Process All Frames ---
frames_data = []

for i, depth_file in enumerate(depth_files):
    ret, color_img = video.read()
    if not ret:
        break
    depth_img = np.load(depth_file)

    patch_depths = [
        depth_img[y, x] * 0.001
        for dx in range(-5, 6)
        for dy in range(-5, 6)
        if 0 <= (x := bbox_center[0] + dx) < 640 and 0 <= (y := bbox_center[1] + dy) < 480
        and mask_resized[y, x] and 0.1 < depth_img[y, x] * 0.001 < 2.0
    ]

    if patch_depths:
        depth_val = np.median(patch_depths)
        X, Y, Z = pixel_to_xyz(*bbox_center, depth_val, intrinsics)
        width_m, height_m = pixel_to_metric(x1 - x0, y1 - y0, depth_val, intrinsics)
    else:
        X = Y = Z = width_m = height_m = depth_val = None

    bg_depths = [depth_img[y0 - 10, x] * 0.001 for x in range(x0, x1) if 0.1 < depth_img[y0 - 10, x] * 0.001 < 2.0]
    thickness_m = max(0, np.mean(bg_depths) - depth_val) if bg_depths and depth_val else None

    # Annotate
    cv2.rectangle(color_img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.circle(color_img, bbox_center, 5, (0, 255, 0), -1)
    lines = [
        f"X: {X:.3f} m, Y: {Y:.3f} m, Z: {Z:.3f} m" if X else "Position: N/A",
        f"Width: {width_m*100:.1f} cm, Height: {height_m*100:.1f} cm" if width_m else "Size: N/A",
        f"Depth: {thickness_m*100:.1f} cm" if thickness_m else "Depth: N/A",
        f"Shape: {shape_type}"
    ]
    for j, txt in enumerate(lines):
        cv2.putText(color_img, txt, (10, 30 + j * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("SAM + Depth Playback", color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frames_data.append((color_img.copy(), depth_img.copy()))

# --- Hold the last frame ---
if frames_data:
    last_color, last_depth = frames_data[-1]
    cv2.imshow("SAM + Depth Playback", last_color)
    print("End of playback. Press any key to close.")
    cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()

# --- Show Point Cloud from Last Frame ---
if frames_data:
    pcd = create_point_cloud(last_color, last_depth, intrinsics)
    o3d.visualization.draw_geometries([pcd])
