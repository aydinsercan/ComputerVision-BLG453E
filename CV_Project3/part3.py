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

    convolution = np.zeros((row_num, colm_num)) 
    reversed_kernel = kernel[::-1, ::-1]    #take time reversal of signal in both dimentions
    for r in range(row_num):
        for c in range(colm_num):
            convolution[r, c] = np.sum(np.multiply(padded_mtx[r:r+kernel_size, c:c+kernel_size], reversed_kernel))

    return convolution


def detect_corner(image, k):
    gaussian_three = np.array([[1,  4, 1], \
                               [4, 12, 4], \
                               [1,  4, 1]])
    gaussian_three = gaussian_three/np.sum(gaussian_three)  #ensure that result will be still in range of [0, 255]

    padded = add_padding(image)
    grad_x = (padded[2:, 1:-1] - padded[:-2, 1:-1])/2   #[I(x+1, y) - I(x-1, y)]/2
    grad_y = (padded[1:-1, 2:] - padded[1:-1, :-2])/2   #[I(x, y+1) - I(x, y-1)]/2

    #In order to create a window traverse effect we must convolve gradients with gaussian kernel
    I_xx = matrix_convolve(grad_x**2, gaussian_three)
    I_xy = matrix_convolve(grad_x*grad_y, gaussian_three)
    I_yy = matrix_convolve(grad_y**2, gaussian_three)

    det_G = I_xx*I_yy - I_xy**2     #since we convolve with gaussian kernel this is not 0 anymore
    trace_G = I_xx + I_yy

    C = det_G - k*trace_G**2

    return C


gaussian_five = np.array([[1,  4,  7,  4, 1], \
                          [4, 16, 26, 16, 4], \
                          [7, 26, 41, 26, 7], \
                          [4, 16, 26, 16, 4], \
                          [1,  4,  7,  4, 1]])
gaussian_five = gaussian_five/np.sum(gaussian_five) #ensure that result will be still in range of [0, 255]


time.sleep(5)

myScreenshot = pyautogui.screenshot()
# myScreenshot.save("ss.png")

shot = np.array(myScreenshot)[:,:,::-1]
cropped = shot[:,:,:]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
smooth_gray = matrix_convolve(gray, gaussian_five)

corner = detect_corner(smooth_gray, 0.01) # k between 0.04 - 0.06
points = (corner > 0)
cropped[points] = [0, 255, 0]

cv2.imshow("image", cropped)
cv2.imwrite("denemeson.png", cropped)
cv2.waitKey()

