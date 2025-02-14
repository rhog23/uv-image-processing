import cv2
import matplotlib.pyplot as plt

img_path = "smaller-image-02.jpg"  # lokasi citra disesuaikan

# opencv
img = cv2.imread(img_path)
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # grayscale
rescv = cv2.bitwise_not(img)

fig, axs = plt.subplots(ncols=2)
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img, cmap="gray")
axs[1].set_title("OpenCV Result")
axs[1].set_axis_off()
axs[1].imshow(rescv, cmap="gray")
plt.show()
