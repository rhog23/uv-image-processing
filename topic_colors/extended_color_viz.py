import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic image
sample_img = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(sample_img, (30, 30), (270, 170), (0, 255, 0), -1)  # green
cv2.circle(sample_img, (150, 100), 50, (0, 0, 255), -1)  # red

# Convert to different color spaces
img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(sample_img, cv2.COLOR_BGR2Lab)
img_ycrcb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2YCrCb)

# Split into channels
r, g, b = cv2.split(img_rgb)
h, s, v = cv2.split(img_hsv)
l, a, b_lab = cv2.split(img_lab)
y, cr, cb = cv2.split(img_ycrcb)

# Plot everything
fig, axs = plt.subplots(4, 4, figsize=(18, 12))

axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("Original (RGB)")
axs[0, 0].axis("off")

axs[0, 1].imshow(r, cmap="Reds")
axs[0, 1].set_title("Red Channel")
axs[0, 1].axis("off")

axs[0, 2].imshow(g, cmap="Greens")
axs[0, 2].set_title("Green Channel")
axs[0, 2].axis("off")

axs[0, 3].imshow(b, cmap="Blues")
axs[0, 3].set_title("Blue Channel")
axs[0, 3].axis("off")

axs[1, 0].imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
axs[1, 0].set_title("HSV Image")
axs[1, 0].axis("off")

axs[1, 1].imshow(h, cmap="hsv")
axs[1, 1].set_title("Hue Channel")
axs[1, 1].axis("off")

axs[1, 2].imshow(s, cmap="gray")
axs[1, 2].set_title("Saturation Channel")
axs[1, 2].axis("off")

axs[1, 3].imshow(v, cmap="gray")
axs[1, 3].set_title("Value Channel")
axs[1, 3].axis("off")

axs[2, 0].imshow(cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB))
axs[2, 0].set_title("Lab Image")
axs[2, 0].axis("off")

axs[2, 1].imshow(l, cmap="gray")
axs[2, 1].set_title("Lightness (L)")
axs[2, 1].axis("off")

axs[2, 2].imshow(a, cmap="gray")
axs[2, 2].set_title("Green-Red (a*)")
axs[2, 2].axis("off")

axs[2, 3].imshow(b_lab, cmap="gray")
axs[2, 3].set_title("Blue-Yellow (b*)")
axs[2, 3].axis("off")

axs[3, 0].imshow(cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB))
axs[3, 0].set_title("YCrCb Image")
axs[3, 0].axis("off")

axs[3, 1].imshow(y, cmap="gray")
axs[3, 1].set_title("Luma (Y)")
axs[3, 1].axis("off")

axs[3, 2].imshow(cr, cmap="gray")
axs[3, 2].set_title("Chroma Red (Cr)")
axs[3, 2].axis("off")

axs[3, 3].imshow(cb, cmap="gray")
axs[3, 3].set_title("Chroma Blue (Cb)")
axs[3, 3].axis("off")

plt.tight_layout()
plt.show()
