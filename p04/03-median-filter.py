import argparse
import cv2
import matplotlib.pyplot as plt


def run(image="", filter_size=3):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = cv2.medianBlur(img, filter_size)

    _, axs = plt.subplots(ncols=2)
    axs[0].set_title("Original Image")
    axs[0].set_axis_off()
    axs[0].imshow(img)
    axs[1].set_title("Result")
    axs[1].set_axis_off()
    axs[1].imshow(result)
    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to image")
    parser.add_argument("-f", "--filter_size", type=int, default=3, help="filter size")
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    try:
        run(**vars(opt))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
