import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../images/daun.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_cpy = img.copy()

try:
    f1 = int(input("Enter minimum value: "))
    f2 = int(input("Enter maximum value: "))

    img_cpy[img >= f2] = 255
    img_cpy[img <= f1] = 0

    fig, axs = plt.subplots(ncols=2)
    axs[0].set_title("Original Image")
    axs[0].set_axis_off()
    axs[0].imshow(img)
    axs[1].set_title(f"Result\nf1: {f1} | f2: {f2}")
    axs[1].set_axis_off()
    axs[1].imshow(img_cpy)
    plt.show()
except:
    print("Error")
