import cv2
import numpy as np
import matplotlib.pyplot as plt

# create a synthetic image (optional, you can use an example image instead)
sample_img = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(sample_img, (30, 30), (270, 170), (0, 255, 0), -1)  # green
cv2.circle(sample_img, (150, 100), 50, (0, 0, 255), -1)  # red

# convert to different color spaces
img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
img_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)  # BGR -> Grayscale
img_hsv = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)  # BGR -> HSV
img_hsl = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HLS)  # BGR -> HLS
img_lab = cv2.cvtColor(sample_img, cv2.COLOR_BGR2LAB)  # BGR -> LAB
img_ycrcb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2YCrCb)  # BGR -> YCrCb

# Plot them
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("RGB")
axs[0, 0].axis("off")

axs[0, 1].imshow(img_gray, cmap="gray")
axs[0, 1].set_title("Grayscale")
axs[0, 1].axis("off")

axs[0, 2].imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
axs[0, 2].set_title("HSV")
axs[0, 2].axis("off")

axs[1, 0].imshow(cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB))
axs[1, 0].set_title("Lab")
axs[1, 0].axis("off")

axs[1, 1].imshow(cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB))
axs[1, 1].set_title("YCrCb")
axs[1, 1].axis("off")

axs[1, 2].imshow(sample_img[..., ::-1])  # BGR to RGB manually
axs[1, 2].set_title("Original (BGR to RGB)")
axs[1, 2].axis("off")

plt.tight_layout()
plt.show()
