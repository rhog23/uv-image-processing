import cv2

img = cv2.imread("smaller-image-02.jpg")

# Perbesar 2x
resized_up = cv2.resize(img, None, fx=2, fy=2)

# Perkecil 0.5x
resized_down = cv2.resize(img, None, fx=0.5, fy=0.5)

cv2.imshow("Upscaled Image", resized_up)
cv2.imshow("Downscaled Image", resized_down)
cv2.waitKey(0)
cv2.destroyAllWindows()
