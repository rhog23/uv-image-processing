import cv2
import sys, time

video_capture = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

aoi_top_left = (200, 75)
aoi_bottom_right = (520, 425)
