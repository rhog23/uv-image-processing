import cv2
import numpy as np


def main():
    # Simulate loading a raw Bayer pattern image (You can replace this with a real Bayer image)
    img_bayer = cv2.imread(
        "bayer_sample.png", cv2.IMREAD_GRAYSCALE
    )  # Should be single channel Bayer
    if img_bayer is None:
        print("Bayer image not found! Make sure 'bayer_sample.png' exists.")
        return

    # Apply Demosaicing
    img_bgr = cv2.cvtColor(img_bayer, cv2.COLOR_BayerBG2BGR)

    # Show both
    cv2.imshow("Raw Bayer Pattern", img_bayer)
    cv2.imshow("Demosaiced BGR Image", img_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
