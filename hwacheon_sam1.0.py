import os
import torch
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from segment_anything import sam_model_registry, SamPredictor

# Step 1: Load the model checkpoint
checkpoint_path = "sam_vit_l_0b3195.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM model
print("Loading SAM model...")
sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)
print("Model loaded.")

# Step 2: Load your image
image_path = "/home/luxolis/work/real-time-3D/hwacheon_3D/manijaCavity/rgb/KakaoTalk_20250528_134341462.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Set the image in the predictor
predictor.set_image(image_rgb)

# Step 4: Define bounding box prompt
h, w, _ = image.shape
bbox = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4])  # x0, y0, x1, y1

# Step 5: Predict mask with timing
start_time = time.time()
masks, scores, _ = predictor.predict(box=bbox[None, :], multimask_output=True)
inference_time = time.time() - start_time
print(f"Mask prediction completed in {inference_time:.4f} seconds.")

best_idx = np.argmax(scores)
mask = masks[best_idx]

# Step 6: Visualize mask (optional)
plt.figure(figsize=(10, 5))
plt.imshow(image_rgb)
plt.imshow(mask, alpha=0.5, cmap='viridis')
plt.title(f"Predicted Mask (Score: {scores[best_idx]:.3f})")
plt.axis("off")
plt.show()

# Step 7: Save mask and masked image
object_mask = (mask.astype(np.uint8)) * 255
masked_image = cv2.bitwise_and(image, image, mask=object_mask)

cv2.imwrite("object_mask.png", object_mask)
cv2.imwrite("object_segmented.png", masked_image)
print("Saved 'object_mask.png' and 'object_segmented.png'.")

# Step 8: Optionally crop the masked object tightly
ys, xs = np.where(mask)
if len(xs) > 0 and len(ys) > 0:
    x0, y0 = np.min(xs), np.min(ys)
    x1, y1 = np.max(xs), np.max(ys)
    cropped = masked_image[y0:y1, x0:x1]
    cv2.imwrite("object_cropped.png", cropped)
    print("Saved cropped object to 'object_cropped.png'.")
else:
    print("Warning: No mask area detected.")
