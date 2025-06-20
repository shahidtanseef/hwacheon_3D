import cv2
import numpy as np
import glob
import os

# Load all depth .npy files sorted
depth_files = sorted(glob.glob("recorded_depth/depth_*.npy"))

# Open the RGB video
video = cv2.VideoCapture("rgb.avi")

frame_index = 0

while video.isOpened() and frame_index < len(depth_files):
    ret, rgb_frame = video.read()
    if not ret:
        print("End of video or read error.")
        break

    # Load corresponding depth frame
    depth_data = np.load(depth_files[frame_index])

    # Normalize depth for visualization
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_data, alpha=0.03),
        cv2.COLORMAP_JET
    )

    # Stack RGB and depth side by side
    combined = np.hstack((rgb_frame, depth_colormap))

    # Display
    cv2.imshow("RGB + Depth", combined)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_index += 1

video.release()
cv2.destroyAllWindows()
