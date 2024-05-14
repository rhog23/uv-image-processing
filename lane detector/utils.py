import typing
import cv2
import numpy as np

COLOR_CHANNELS_CONVERSION: typing.Dict = {
    "gray": cv2.COLOR_BGR2GRAY,
    "rgb": cv2.COLOR_BGR2RGB,
    "hsv": cv2.COLOR_BGR2HSV,
    "hsl": cv2.COLOR_BGR2HLS,
}


def _empty_trackball_callback(_null):
    pass


def init_trackball(init_val, win_name, target_width=240, target_height=180):
    cv2.namedWindow(win_name)
    # cv2.resizeWindow(win_name, 360, 240)
    cv2.createTrackbar(
        "Top LR", win_name, init_val[0], target_width, _empty_trackball_callback
    )

    cv2.createTrackbar(
        "Top UD",
        win_name,
        init_val[1],
        target_height,
        _empty_trackball_callback,
    )

    cv2.createTrackbar(
        "Bottom LR",
        win_name,
        init_val[2],
        target_width,
        _empty_trackball_callback,
    )

    cv2.createTrackbar(
        "Bottom UD",
        win_name,
        init_val[3],
        target_height,
        _empty_trackball_callback,
    )


def draw_wrap_points(frame, points):
    for x in range(4):
        cv2.circle(
            frame,
            (int(points[x][0]), int(points[x][1])),
            5,
            (0, 0, 0),
            cv2.FILLED,
        )

    return frame


def get_histogram(frame, min_per=0.1, display=False, region=1):
    if region == 1:
        hist_values = np.sum(frame, axis=0)
    else:
        hist_values = np.sum(frame[frame.shape[0] // region :, :], axis=0)

    max_value = np.max(hist_values)
    min_value = min_per * max_value

    idx_array = np.where(hist_values >= min_value)
    base_point = int(np.average(idx_array))

    if display:
        frame_hist = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

        for x, intensity in enumerate(hist_values):
            cv2.line(
                frame_hist,
                (x, frame.shape[0]),
                (x, frame.shape[0] - intensity // 255 // region),
                (255, 0, 255),
                1,
            )

            cv2.circle(
                frame_hist, (base_point, frame.shape[0]), 20, (0, 255, 255), cv2.FILLED
            )
        return base_point, frame_hist

    return base_point


def wrap_frame(frame, points, width, height, invert=False):
    source_points = np.float32(points)
    target_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    if invert:
        # swap source points and target points
        matrix = cv2.getPerspectiveTransform(target_points, source_points)
    else:
        matrix = cv2.getPerspectiveTransform(source_points, target_points)

    frame_wrap = cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
    )

    return frame_wrap


def thresholding(img):
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0, 0, 201])
    upperWhite = np.array([179, 110, 255])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


"""
def thresholding1(img):
    # otsu thresholding
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # removing illumination noise by selecting colors with high lightness value
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    imgHls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    lowerWhite = np.array([0,255,0])
    upperWhite = np.array([179,255,255])
    maskWhite = cv2.inRange(imgHls,lowerWhite,upperWhite)
    return otsu - maskWhite
    """


def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar(
        "Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing
    )
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)


def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32(
        [
            (widthTop, heightTop),
            (wT - widthTop, heightTop),
            (widthBottom, heightBottom),
            (wT - widthBottom, heightBottom),
        ]
    )
    return points


def drawPoints(img, points):
    for x in range(4):
        cv2.circle(
            img, (int(points[x][0]), int(points[x][1])), 15, (255, 0, 0), cv2.FILLED
        )
    return img


def getHistogram(img, display=False, minPer=0.1, region=1):
    h, w = img.shape[:2]
    histValues = np.sum(img[-h // region :, :], axis=0)

    maxValue = np.max(histValues)  # FIND THE MAX VALUE
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))

    if display:
        imgHist = np.zeros((h, w, 3), np.uint8)
        for x, intensity in enumerate(histValues):
            # print(intensity)
            if intensity > minValue:
                color = (211, 211, 211)
            else:
                color = (200, 165, 200)
            cv2.line(
                imgHist, (x, h), (x, int(h - (intensity // region // 255))), color, 1
            )
            cv2.circle(imgHist, (basePoint, h), 20, (255, 200, 0), cv2.FILLED)
        return basePoint, imgHist

    return basePoint


def stackImages(scale, imgArray):

    if isinstance(imgArray[0], list):
        return imageMatrix(scale, imgArray)
    return imageArray(scale, imgArray)


def imageMatrix(scale, imgArray):
    rows, cols = len(imgArray), len(imgArray[0])

    hImg, wImg = imgArray[0][0].shape[:2]

    for x in range(0, rows):
        for y in range(0, cols):
            imgArray[x][y] = cv2.resize(
                imgArray[x][y], (int(wImg * scale), int(hImg * scale)), None
            )
            if len(imgArray[x][y].shape) == 2:
                imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

    imageBlank = np.zeros((hImg, wImg, 3), np.uint8)
    hor = [imageBlank] * rows
    for x in range(0, rows):
        hor[x] = np.hstack(imgArray[x])

    return np.vstack(hor)


def imageArray(scale, imgArray):

    for x in range(0, len(imgArray)):

        imgArray[x] = cv2.resize(
            imgArray[x],
            (imgArray[0].shape[1], imgArray[0].shape[0]),
            None,
            scale,
            scale,
        )

        if len(imgArray[x].shape) == 2:
            imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

    return np.hstack(imgArray)


def stopDetector(img, cascadePath, minArea):
    cascade = cv2.CascadeClassifier(cascadePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaleVal = 1 + 250 / 1000
    neig = 12
    objects = cascade.detectMultiScale(gray, scaleVal, neig)
    for x, y, w, h in objects:
        area = w * h
        if area > minArea:
            return w
    return 0


def distance_to_camera(reflectedWidth):
    # https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    # compute and return the distance from the marker to the camera
    knownWidth = 4.5  # width stop sign
    focalLength = 330  # mm to be obtained
    """
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)"""
    return (knownWidth * focalLength) / reflectedWidth
