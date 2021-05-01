
# SERHAT DEMİRKIRAN
# SERCAN AYDIN

import time, pyautogui, os
import cv2 as cv
import numpy as np

def get_counters(circles):
    counter = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            counter += 1
        return counter
    return 0

def func_val(image_A, image_S, image_D):
    A=0
    S=0
    D=0
    Counter_ = 50

    while True:
        counter_A = get_counters(cv.HoughCircles(image_A, cv.HOUGH_GRADIENT, 1, image_A.shape[0] / 32, param1=100, param2=30, minRadius=0, maxRadius=0))
        counter_S = get_counters(cv.HoughCircles(image_S, cv.HOUGH_GRADIENT, 1, image_S.shape[0] / 32, param1=100, param2=30, minRadius=0, maxRadius=0))
        counter_D = get_counters(cv.HoughCircles(image_D, cv.HOUGH_GRADIENT, 1, image_D.shape[0] / 32, param1=100, param2=30, minRadius=0, maxRadius=0))

        if counter_A > counter_S and counter_A > counter_D:
            A += 1
        elif counter_S > counter_D:
            S += 1
        else:
            D += 1

        if A == Counter_:
            return pyautogui.press('a')
        if S == Counter_:
            return pyautogui.press('s')
        if D == Counter_:
            return pyautogui.press('d')

time.sleep(3)
while True:
    image_name = "Ss.png"
    my_screenshot = pyautogui.screenshot()
    my_screenshot.save(image_name)
    image = cv.imread(image_name)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_A = image[150:550, 140:540]
    image_S = image[150:550, 740:1140]
    image_D = image[150:550, 1340:1740]
    func_val(image_A, image_S, image_D)


# We used HoughCircles transform to find circles in an given screenshot.
# Therefore, we calculated the dots on these dices.

