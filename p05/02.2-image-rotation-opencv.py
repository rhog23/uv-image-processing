import cv2
import matplotlib.pyplot as plt
import argparse


def run(image="", angle="", coord=tuple()):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    (h, w) = img.shape[:2]

    if not coord:
        coord = (w // 2, h // 2)
    else:
        coord = [int(c) for c in coord]

    m = cv2.getRotationMatrix2D(tuple(coord), angle, 1.0)
    result = cv2.warpAffine(img, m, (w, h))

    plt.figure()
    plt.subplot(121)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(img)
    plt.subplot(122)
    plt.title(f"Result | Angle: {angle}°")
    plt.axis("off")
    plt.imshow(result)
    plt.show()


def main(opt):
    run(**vars(opt))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to image", required=True)
    parser.add_argument(
        "-a",
        "--angle",
        type=int,
        help="angle of rotation, can be positive or negative",
        required=True,
    )
    parser.add_argument(
        "-c", "--coord", nargs="+", help="coodinate of rotation in the form of x,y"
    )
    opt = parser.parse_args()
    print(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
