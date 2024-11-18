import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

capture = cv.VideoCapture('Person.mp4')

while True:
    isTrue, frame = capture.read()

    resize = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv.INTER_CUBIC)

    color_HSV = cv.cvtColor(resize, cv.COLOR_BGR2HSV)

    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

    eroded = cv.erode(thresh, (9, 9), iterations=7)

    contours, hierarchy = cv.findContours(eroded, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    frame_ct = cv.drawContours(color_HSV, contours, -1, (255, 255, 0), 2)

    cv.imshow("HSV", frame_ct)

    min_contour_area = 3000
    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv.boundingRect(cnt)
        frame_out = cv.rectangle(resize, (x, y), (x + w, y + h), (255, 255, 0), 3)

    cv.imshow("Person", frame_out)
    
    if cv.waitKey(20) and 0xFF == ord('d'):
        break

capture.release
cv.destroyAllWindows()