import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.color import label2rgb

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can't open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    gray = color.rgb2gray(frame)

    cv2.imshow("frame", image)

    if cv2.waitKey(1) == ord("q"):
        break

# apply threshold
#
# bw = closing(image > thresh, disk(3))
# bw = np.bitwise_not(bw)
# cleared = clear_border(bw)
# label_image = label(cleared)
# image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
# for region in regionprops(label_image):
#     print(region.area)
#     print(region.perimeter)
cap.release()
cv2.destroyAllWindows()
