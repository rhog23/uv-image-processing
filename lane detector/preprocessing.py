import cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 160)
cap.set(4, 120)

last_lanes = [-999, -999]


def canny(frame):
    blur = cv2.GaussianBlur(
        frame,
        (
            5,
            5,
        ),
        0,
    )
    return cv2.Canny(blur, 125, 200)


def add_mask(frame):
    h, w = frame.shape[:2]
    triangle = np.array(
        [
            [
                (int(10), int(6 * h / 7)),
                (int(w - 10), int(6 * h / 7)),
                (int(w - 10), int(5 * h / 7)),
                (int(10), int(5 * h / 7)),
            ]
        ]
    )

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, triangle, 255)
    frame = cv2.bitwise_and(frame, mask)
    return frame


def find_edges(frame):
    # line_frame = np.zeros_like(frame)
    edges = []

    np_frame = np.array(frame)

    (x, y) = np.nonzero(np_frame)
    temp_edges = list(zip(y, x))

    if temp_edges != []:
        cluster = linkage(temp_edges, method="single", metric="euclidean")
        group = fcluster(cluster, 5, criterion="distance")
        edges = [[[], []] for idx in range(max(group))]

        for i in range(0, len(temp_edges)):
            edges[group[i] - 1][0] += [temp_edges[i][0]]  # x array
            edges[group[i] - 1][1] += [temp_edges[i][1]]  # y array

    return edges


def curved_lines(frame, edges, plot=False):
    line_frame = np.zeros_like(frame)
    h, w = frame.shape[:2]
    x_avg = -1
    phi_avg = -1
    lanes = []
    lane1 = [-1, -1, 0]
    lane2 = [-1, -1, 0]

    for idx in range(0, len(edges)):
        coeff = np.polyfit(
            edges[idx][1], edges[idx][0], 2
        )  # fit quadratic equation to group of points to estimate lane curvature
        existing_lane = False

        if (
            len(edges[idx][0]) > 50
        ):  # lane lines will have at least 50 points in worst case - anything shorter will be ignored
            if plot:
                y = np.linspace(h / 2, h, 5)
                x = np.polyval(coeff, y)
                points = (np.asarray([x, y]).T).astype(np.int32)
                cv2.polylines(line_frame, [points], False, (255, 255, 255), thickness=3)

            x_line = (
                coeff[0] * (4 / 3) * h**2 + coeff[1] * (4 / 3) * h + coeff[2]
            )  # find x intercept at "center" of rover
            phi = (
                2 * coeff[0] * (6 * h / 7) + coeff[1]
            )  # find slope at 1/7th height up frame to anticipate curvature from first derivative of polynomial

            for i in lanes:  # group lines with similar x intercepts together
                if (x_line) - 100 <= i[0] / i[2] <= (x_line + 100):
                    i[0] += x_line
                    i[1] += phi
                    i[2] += 1
                    existing_lane = True
                    break

            if (
                existing_lane == False
            ):  # if the lane x_intercept isn't close to any other existing lane in lanes array
                lanes += [[x_line, phi, 1]]

    for i in lanes:
        if (
            i[2] > 1
        ):  # lanes will have at least 2 lines combined from above, anything less than that can be ignored
            cv2.line(
                line_frame,
                (within_frame(i[0] / i[2], w), h),
                (within_frame(i[0] / i[2], w), 9 * h / 10),
                (0, 255, 0),
                3,
            )

            if i[2] > lane1[2]:
                lane2 = lane1
                lane1 = [i[0] / i[2], i[1] / i[2], i[2]]

            elif i[2] > lane2[2]:
                lane2 = [i[0] / i[2], i[1] / i[2], i[2]]

    if lane1[0] != -1 and lane2[0] != -1:
        x_avg = (lane1[0] + lane2[0]) / 2  # take average of right and left lines
        phi_avg = (
            -1 * np.arctan((lane1[1] + lane2[1]) / 2) * 180 / np.pi
        )  # find angle in rads of slope
        cv2.line(
            line_frame,
            (within_frame(x_avg, w), h),
            (
                within_frame((x_avg - ((lane1[1] + lane2[1]) / 2) * h / 5), w),
                4 * h / 5,
            ),
            (0, 0, 255),
            3,
        )
        last_lanes = [lane1[0], lane2[0]]

    cv2.putText(
        line_frame,
        ("x error :" + str(x_avg - 320)),
        (0, 0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
        False,
    )
    cv2.putText(
        line_frame,
        ("phi error :" + str(phi_avg)),
        (0, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
        False,
    )

    return [line_frame, x_avg, phi_avg]


def within_frame(index, w):
    return int(min(w, max(0, index)))


while True:
    try:
        ret, frame = cap.read()

        # rotate frame
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        r_canny = canny(frame)
        # add mask
        r_masked = add_mask(r_canny)

        r_edges = find_edges(r_masked)

        [cv_lines, x_error, phi_error] = curved_lines(frame, r_edges, True)
        final = cv2.addWeighted(cv_lines, 0.8, frame, 1, 1)

        cv2.imshow("frame", final)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
