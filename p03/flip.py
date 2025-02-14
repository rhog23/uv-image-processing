import cv2

img = cv2.imread("smaller-image-02.jpg")

# Flip horizontal
flipped_hor = cv2.flip(img, 1)

# Flip vertikal
flipped_ver = cv2.flip(img, 0)

# Flip keduanya
flipped_both = cv2.flip(img, -1)

cv2.imshow("Horizontal Flip", flipped_hor)
cv2.imshow("Vertical Flip", flipped_ver)
cv2.imshow("Both Flip", flipped_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
