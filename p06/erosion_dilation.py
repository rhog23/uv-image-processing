import argparse
import sys
import cv2
import functools

max_se: int = (
    2  # variabel yang menyimpan banyaknya jenis structuring element (se). di dalam kasus ini ada 3 jenis (0, 1, 2)
)
max_kernel_size: int = (
    20  # variabel yang menyimpan jenis dari kernel. di dalam kasus ini ada 10 jenis. Ukuran dari kernel selalu bernilai ganjil
)
title_trackbar_element_shape: str = (
    "Element"  # nama dari trackbar yang digunakan untuk memilih jenis se
)
title_trackbar_kernel_size: str = (
    "Kernel size"  # nama dari trackbar yang digunakan untuk memilih ukuran kernel
)
title_erosion_window: str = (
    "Erosion Demo"  # nama dari jendela yang digunakan untuk menampilkan hasil erosi
)
title_dilation_window: str = (
    "Dilation Demo"  # nama dari jendela yang digunakan untuk menampilkan hasil dilasi
)


def structuring_element(shape: int, size: int) -> cv2.typing.MatLike:
    """Fungsi yang digunakan untuk menghasilkan structuring element.
    Structuring element adalah sebuah kernel yang digunakan untuk melakukan operasi morfologi.


    Args:
        shape (int): Jenis structuring element. Berikut ini adalah tipe-tipenya: `tipe 0: rectangle | tipe 1: cross | tipe 2: ellipse (lingkaran)`
        size (int): Ukuran dari kernel

    Returns:
        cv2.typing.MatLike: structuring element
    """
    shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]
    return cv2.getStructuringElement(shapes[shape], (size, size))


def erosion(val: int, src: cv2.typing.MatLike) -> None:
    """Fungsi untuk menghasilkan efek erosion

    Args:
        val (int): parameter default yang dibutuhkan oleh OpenCV untuk mengambil value trackbar. Tidak dipanggil di dalam fungsi
        src (cv2.typing.MatLike): citra
    """
    # print(
    #     f"Current trackbar value: {val}"
    # )  # Uncomment untuk melihat trackking nilai dari variabel val

    # tipe structuring element
    se_type = cv2.getTrackbarPos(title_trackbar_element_shape, title_erosion_window)

    # ukuran structuring element
    se_size = (
        cv2.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window) + 1
    )  # harus ditambah 1 agar tidak error (error karena ukuran kernel tidak boleh 0)

    element = structuring_element(
        se_type,
        se_size,
    )  # ukuran dari structuring element sebaiknya selalu ganjil (1, 3, 5, dst.)

    result = cv2.erode(
        src, element
    )  # mengaplikasikan erosi pada gambar dengan menggunakan structuring element yang telah dibuat

    cv2.imshow(title_erosion_window, result)  # menampilkan hasil


def dilation(val: int, src: cv2.typing.MatLike) -> None:
    """Fungsi untuk menghasilkan efek dilation

    Args:
        val (int): parameter default yang dibutuhkan oleh OpenCV untuk mengambil value trackbar. Tidak dipanggil di dalam fungsi
        src (cv2.typing.MatLike): citra
    """
    # tipe structuring element
    se_type = cv2.getTrackbarPos(title_trackbar_element_shape, title_dilation_window)

    # ukuran structuring element
    se_size = (
        2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window) + 1
    )  # harus ditambah 1 agar tidak error (error karena ukuran kernel tidak boleh 0)

    element = structuring_element(
        se_type,
        se_size,
    )  # ukuran dari structuring element sebaiknya ganjil (1, 3, 5, dst.)

    result = cv2.dilate(
        src, element
    )  # mengaplikasikan dilation pada gambar dengan menggunakan structuring element yang telah dibuat
    cv2.imshow(title_dilation_window, result)


def main(image_path):
    try:

        src = cv2.imread(image_path)  # load citra / gambar
        src = cv2.resize(
            src, (720, 480)
        )  # diubah ke ukuran tertentu dengan format (width, height) -> opsional

        # Erosion window
        cv2.namedWindow(title_erosion_window)
        # Menampilkan trackbar tipe SE di dalam erosion window
        cv2.createTrackbar(
            title_trackbar_element_shape,
            title_erosion_window,
            0,
            max_se,
            functools.partial(erosion, src=src),
        )
        # Menampilkan trackbar ukuran SE (kernel) di dalam erosion window
        cv2.createTrackbar(
            title_trackbar_kernel_size,
            title_erosion_window,
            0,
            max_kernel_size,
            functools.partial(erosion, src=src),
        )

        # Dilation window
        cv2.namedWindow(title_dilation_window)
        # Menampilkan trackbar tipe SE di dalam dilation window
        cv2.createTrackbar(
            title_trackbar_element_shape,
            title_dilation_window,
            0,
            max_se,
            functools.partial(dilation, src=src),
        )

        # Menampilkan trackbar ukuran SE (kernel) di dalam dilation window
        cv2.createTrackbar(
            title_trackbar_kernel_size,
            title_dilation_window,
            0,
            max_kernel_size,
            functools.partial(dilation, src=src),
        )

        # Initial display
        erosion(1, src)
        dilation(1, src)

        cv2.waitKey()
    except Exception as e:
        print(f"!!!Terjadi kesalahan!!!\nINFO: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    main(args.image)
