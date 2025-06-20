import os
import torch
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

# Step 1: Download the model checkpoint (if not already downloaded)
#checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
checkpoint_path = "sam_vit_l_0b3195.pth"

# Step 2: Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# Step 3: Load an image (replace with your own)
image_path = "/home/luxolis/work/real-time-3D/hwacheon_3D/manijaCavity/rgb/KakaoTalk_20250528_134341462.jpg"  # <-- Replace this with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 4: Set the image for the predictor
predictor.set_image(image_rgb)

# Step 5: Define bounding box (x0, y0, x1, y1) manually or from estimate
# Example: manually define box around center area
h, w, _ = image.shape
bbox = np.array([w//4, h//4, 3*w//4, 3*h//4])  # replace with your estimate

# Step 6: Predict mask
masks, scores, _ = predictor.predict(box=bbox[None, :], multimask_output=True)

# Step 7: Visualize the best mask
best_idx = np.argmax(scores)
mask = masks[best_idx]

# Display
plt.figure(figsize=(10, 5))
plt.imshow(image_rgb)
plt.imshow(mask, alpha=0.5, cmap='viridis')
plt.title(f"Predicted Mask (Score: {scores[best_idx]:.3f})")
plt.axis("off")
plt.show()
