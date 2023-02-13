import argparse
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def run(image="", filter_size=3):
    img = Image.open(image)

    # apply min filter
    min_img = img.filter(ImageFilter.MinFilter(size=filter_size))

    # apply max filter
    max_img = img.filter(ImageFilter.MaxFilter(size=filter_size))

    fig, axs = plt.subplots(ncols=3)
    axs[0].set_title("Original Image")
    axs[0].set_axis_off()
    axs[0].imshow(img, cmap="gray")
    axs[1].set_title("Min Image")
    axs[1].set_axis_off()
    axs[1].imshow(min_img, cmap="gray")
    axs[2].set_title("Max Image")
    axs[2].set_axis_off()
    axs[2].imshow(max_img, cmap="gray")
    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to image")
    parser.add_argument(
        "-f",
        "--filter_size",
        default=3,
        type=int,
        help="filter size, must be odd number and greater or equals to 3",
    )
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
