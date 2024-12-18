import argparse
import cv2
import matplotlib.pyplot as plt


def run(image=""):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    while True:
        print("Resizing Type:\na: percentage\nb: fixed\nq: quit")
        resize_type = input("Enter resizing type: ").lower()

        if resize_type == "a":
            print(f"{'='*24} Percentage {'='*24}")
            print("percentage < 100% will downscale the image")
            print("percentage = 100% doesn't change the image size")
            print("percentage > 100% will upscales the image")
            print(f"{'='*60}")
            try:
                percentage = int(input("Enter resizing percentage (integer): "))
                width = int(img.shape[1] * percentage / 100)
                height = int(img.shape[0] * percentage / 100)
                dimension = (width, height)

                result = cv2.resize(img, dimension)
                print(
                    f"Original Image Shape (width * height): {img.shape[1]} * {img.shape[0]}"
                )
                print(
                    f"Resulting Image Shape (width * height): {result.shape[1]} * {result.shape[0]}"
                )
                show_result(img, result)
                return False
            except Exception as e:
                print(f"Error: {e}")

        if resize_type == "b":
            print(f"{'='*10} Fixed {'='*10}")
            try:
                width = int(input("Enter width: "))
                height = int(input("Enter heigt: "))

                if width == 0:
                    width = img.shape[1]

                if height == 0:
                    height = img.shape[0]

                dimension = (width, height)

                result = cv2.resize(img, dimension)
                print(
                    f"Original Image Shape (width * height): {img.shape[1]} * {img.shape[0]}"
                )
                print(
                    f"Resulting Image Shape (width * height): {result.shape[1]} * {result.shape[0]}"
                )
                show_result(img, result)
                return False
            except Exception as e:
                print(f"Error: {e}")

        if resize_type == "q":
            return False

        print("Invalid input. Please enter resizing type")


def show_result(orig_img, result):
    plt.figure()
    plt.subplot(121)
    plt.title("Original Image", fontsize=24)
    plt.imshow(orig_img)
    plt.subplot(122)
    plt.title("Result", fontsize=24)
    plt.imshow(result)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to image")
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
