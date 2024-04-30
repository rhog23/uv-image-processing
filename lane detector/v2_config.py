################################################################################
######## LANE DETECTION PROJECT ################################################
################################################################################
# BY:           CAN OZCIVELEK
# DATE:         DECEMBER 2018
#
# DESCRIPTION:  THIS  PROJECT WAS CREATED  TO DEMONSTRATE HOW  A  LANE DETECTION
#               SYSTEM WORKS  ON CARS EQUIPPED WITH A FRONT  FACING CAMERA. WITH
#               THE HELP OF OPENCV LIBRARIES IT IS POSSIBLE TO DESIGN ALGORITHMS
#               THAT CAN  IDENTIFY LANE LINES, AND PREDICT STEERING ANGLES, ALSO
#               WARN  DRIVERS  IF THE CAR IS  DRIFTING  AWAY FROM  CURRENT LANE.
################################################################################


# IMPORT NECESSARY LIBRARIES
import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

# Defining variables to hold meter-to-pixel conversion
ym_per_pix = 30 / 360
# Standard lane width is 3.7 meters divided by lane width in pixels which is
# calculated to be approximately 720 pixels not to be confused with frame height
xm_per_pix = 3.7 / 360

# Get path to the current working directory
CWD_PATH = os.getcwd()


def readVideo(src: str | int = 0):

    if src != 0:
        cap = cv2.VideoCapture(os.path.join(CWD_PATH, src))
    else:
        cap = cv2.VideoCapture(0)

    return cap


def processImage(
    frame: cv2.typing.MatLike,
    blur_kernel: int = 3,
):

    # # Apply HLS color filtering to filter out white lane lines
    # hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    # lower_white = np.array([0, 160, 10])
    # upper_white = np.array([255, 255, 255])
    # mask = cv2.inRange(inpImage, lower_white, upper_white)
    # hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask)

    # # Convert image to grayscale, apply threshold, blur & extract edges
    # gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    # canny = cv2.Canny(blur, 40, 60)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, tuple([blur_kernel, blur_kernel]), 0)
    thresh = cv2.inRange(blurred_frame, 0, 70)

    return thresh


def perspectiveWarp(inpImage):

    img_size = (inpImage.shape[1], inpImage.shape[0])

    # Perspective points to be warped
    src = np.float32([[295, 220], [345, 220], [100, 320], [500, 320]])

    for p in src:
        cv2.circle(inpImage, tuple([int(x) for x in p]), 5, (0, 0, 0), cv2.FILLED)

    # Window to be shown
    dst = np.float32([[100, 0], [600, 0], [100, 355], [600, 355]])

    for p in dst:
        cv2.circle(inpImage, tuple([int(x) for x in p]), 5, (255, 0, 0), cv2.FILLED)

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

    return birdseye, minv


def plotHistogram(inpImage):

    histogram = np.sum(inpImage[inpImage.shape[0] // 2 :, :], axis=0)

    return histogram


def slide_window_search(binary_warped, histogram):

    # Find the start of left and right lane lines using histogram info
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # A total of 9 windows will be used
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(
            out_img,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            out_img,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (0, 255, 0),
            2,
        )
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Apply 2nd degree polynomial fit to fit curves
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return ploty, left_fit, right_fit, ltx, rtx


def general_search(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
        nonzerox
        > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)
    ) & (
        nonzerox
        < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)
    )

    right_lane_inds = (
        nonzerox
        > (
            right_fit[0] * (nonzeroy**2)
            + right_fit[1] * nonzeroy
            + right_fit[2]
            - margin
        )
    ) & (
        nonzerox
        < (
            right_fit[0] * (nonzeroy**2)
            + right_fit[1] * nonzeroy
            + right_fit[2]
            + margin
        )
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))]
    )
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))]
    )
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))]
    )
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int32([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int32([right_line_pts]), (0, 255, 0))

    ret = {}
    ret["leftx"] = leftx
    ret["rightx"] = rightx
    ret["left_fitx"] = left_fitx
    ret["right_fitx"] = right_fitx
    ret["ploty"] = ploty

    return ret


def measure_lane_curvature(ploty, leftx, rightx):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = (
        (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * left_fit_cr[0])
    right_curverad = (
        (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    # Decide if it is a left or a right curve
    if leftx[0] - leftx[-1] > 60:
        curve_direction = "Left Curve"
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = "Right Curve"
    else:
        curve_direction = "Straight"

    return (left_curverad + right_curverad) / 2.0, curve_direction


def draw_lane_lines(original_image, warped_image, Minv, draw_info):

    left_fitx = draw_info["left_fitx"]
    right_fitx = draw_info["right_fitx"]
    ploty = draw_info["ploty"]

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int32([pts]), (255, 255, 0))
    cv2.fillPoly(color_warp, np.int32([pts_mean]), (0, 255, 255))

    newwarp = cv2.warpPerspective(
        color_warp, Minv, (original_image.shape[1], original_image.shape[0])
    )
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return pts_mean, result


def offCenter(meanPts, inpFrame):

    # Calculating deviation in meters
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction


def addText(img, radius, direction, deviation, devDirection):

    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_TRIPLEX

    if direction != "Straight":
        text = "Radius of Curvature: " + "{:04.0f}".format(radius) + "m"
        text1 = "Curve Direction: " + (direction)

    else:
        text = "Radius of Curvature: " + "N/A"
        text1 = "Curve Direction: " + (direction)

    cv2.putText(img, text, (50, 100), font, 0.5, (0, 100, 200), 1, cv2.LINE_AA)
    cv2.putText(img, text1, (50, 150), font, 0.5, (0, 100, 200), 1, cv2.LINE_AA)

    # Deviation
    deviation_text = (
        "Off Center: " + str(round(abs(deviation), 3)) + "m" + " to the " + devDirection
    )
    cv2.putText(
        img,
        deviation_text,
        (50, 200),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.5,
        (0, 100, 200),
        1,
        cv2.LINE_AA,
    )

    return img


if __name__ == "__main__":

    # Read the input image
    image = readVideo()

    while True:

        _, frame = image.read()
        frame = cv2.resize(frame, (640, 360))

        birdView, minverse = perspectiveWarp(frame)

        thresh = processImage(birdView)

        hist = plotHistogram(thresh)

        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(
            thresh, hist
        )

        draw_info = general_search(thresh, left_fit, right_fit)

        curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)

        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)

        deviation, directionDev = offCenter(meanPts, frame)

        # Adding text to our final image
        finalImg = addText(result, curveRad, curveDir, deviation, directionDev)

        # Displaying final image
        cv2.imshow("Final", finalImg)

        # Wait for the ENTER key to be pressed to stop playback
        if cv2.waitKey(1) == ord("q"):
            break

        if image.get(cv2.CAP_PROP_POS_FRAMES) == image.get(cv2.CAP_PROP_FRAME_COUNT):
            image.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Cleanup
    image.release()
    cv2.destroyAllWindows()
