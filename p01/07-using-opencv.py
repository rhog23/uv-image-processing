import cv2

im = cv2.imread("../images/1.png")
im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
print(f"{'Dimension':<20}:{im.ndim}")
print(f"{'Shape':<20}:{im.shape}")
print(f"{'Height':<20}:{im.shape[0]}")
print(f"{'Width':<20}:{im.shape[1]}")
cv2.imshow("Example Image", im_bgr)
cv2.waitKey(0)
