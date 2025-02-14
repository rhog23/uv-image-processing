import cv2, time
import matplotlib.pyplot as plt

img_path = "white-gundam-thumb.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

try:
    alpha = float(input("Enter a value for alpha (contrast): "))
    beta = int(input("Enter a value for beta (brightness): "))

    start = time.time()
    img_bright_cv = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    end = time.time()

    print(f"Processing time: {end-start:.5f} s")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
    axs[0].set_title("Original Image", fontsize=10)
    axs[0].imshow(img)
    axs[0].set_axis_off()

    axs[1].set_title(
        f"Image Brightness OpenCV\nAlpha:{alpha} | Beta:{beta}", fontsize=10
    )
    axs[1].imshow(img_bright_cv)
    axs[1].set_axis_off()

    plt.show()
except ValueError:
    print("Error, not a number")
