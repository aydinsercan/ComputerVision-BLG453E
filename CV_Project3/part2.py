import pyautogui
import time
import cv2
import numpy as np


def add_padding(matrix):
    new_matrix = np.zeros((matrix.shape[0]+2, matrix.shape[1]+2))
    new_matrix[1:-1, 1:-1] = matrix.copy()
    return new_matrix


def matrix_convolve(matrix, kernel):

    row_num, colm_num = matrix.shape
    kernel_size = kernel.shape[0]

    padded_mtx = matrix
    for i in range(int(kernel_size/2)):
        padded_mtx = add_padding(padded_mtx)    # pad matrix with 1 layer of  0's

    convolution = np.zeros((row_num, colm_num)) #result of the convolution stored in here
    reversed_kernel = kernel[::-1, ::-1]    #take time reversal of signal in both dimentions
    for r in range(row_num):
        for c in range(colm_num):
            convolution[r, c] = np.sum(np.multiply(padded_mtx[r:r+kernel_size, c:c+kernel_size], reversed_kernel))

    return convolution


sobel_x = np.array([[-1, 0, 1], \
                    [-2, 0, 2], \
                    [-1, 0, 1]]) / 4  #ensure that result will be still in range of [0, 255]

sobel_y = np.array([[-1, -2, -1], \
                    [ 0,  0,  0], \
                    [ 1,  2,  1]]) / 4 #ensure that result will be still in range of [0, 255]


gaussian_three = np.array([[1,  4, 1], \
                           [4, 12, 4], \
                           [1,  4, 1]])
gaussian_three = gaussian_three/np.sum(gaussian_three)  #ensure that result will be still in range of [0, 255]

gaussian_five = np.array([[1,  4,  7,  4, 1], \
                          [4, 16, 26, 16, 4], \
                          [7, 26, 41, 26, 7], \
                          [4, 16, 26, 16, 4], \
                          [1,  4,  7,  4, 1]])
gaussian_five = gaussian_five/np.sum(gaussian_five) #ensure that result will be still in range of [0, 255]



time.sleep(5)
#in this 5 seconds you should switch to game screen to transfer the simulated keyboard inputs to the game.

myScreenshot = pyautogui.screenshot()
myScreenshot.save("ss.png")
#And example screenshot is obtained. We will work on screenshots like this for this homework

shot = np.array(myScreenshot)[:,:,::-1]

cropped = shot[:,:,:]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

smooth = np.uint8(matrix_convolve(gray, gaussian_five))    #filter with Gauessian
edges = cv2.Canny(smooth, 200, 250)

cv2.imshow("Screenshot", cropped)
cv2.imshow("Edges", edges)
cv2.imwrite("canny.png", edges)
cv2.waitKey()
