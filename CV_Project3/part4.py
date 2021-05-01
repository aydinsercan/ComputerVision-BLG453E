import pyautogui
import time
import cv2
import numpy as np
import time


while True:
	myScreenshot = pyautogui.screenshot()
	#myScreenshot.save("ss.png")
	shot = np.array(myScreenshot)[:,:,::-1]
	cropped = shot[840:1075,750:1290]
	gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
	# find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

	if len(corners)-1 == 3:
		pyautogui.keyDown('a')
		time.sleep(0.2)
		pyautogui.keyUp('a')
	elif len(corners)-1 == 4:
		pyautogui.keyDown('s') 
		time.sleep(0.2)
		pyautogui.keyUp('s')
	elif len(corners)-1 == 10:
		pyautogui.keyDown('d')
		time.sleep(0.2)
		pyautogui.keyUp('d')
	elif len(corners)-1 == 6:
		pyautogui.keyDown('f')
		time.sleep(0.2)
		pyautogui.keyUp('f')
	else:
		time.sleep(0.2)
		pyautogui.keyUp('a')
		pyautogui.keyUp('s')
		pyautogui.keyUp('d')
		pyautogui.keyUp('f')




"""
path = "C:/Users/SERCAN/Desktop/part"

k_list = []
for i in range(18):
	filename = path + "/shape_" + str(i+1) + ".PNG"
	img = cv2.imread(filename)
	# cv2.imshow("image",img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
	if len(corners)-1 == 3:
		k_list.append('A')
	elif len(corners)-1 == 4:
		k_list.append('S')
	else:
		print("Error")


print(k_list)
#[--3--'A'--5--'S'--5--'A'----'S', 'A', 'A', 'A', 'A', 
# 'S', 'S', 'S', 'A', 'A', 'A', 'S', 'S', 'S', 'S']

time.sleep(2) #initial time
pyautogui.click(path + "/Vabank.PNG")

count = 0
while True:
	if pyautogui.pixel(661,672)[0] == 0:
		if k_list[count] == 'A':
			pyautogui.press('a')
		elif k_list[count] == 'S':
			pyautogui.press('s')
		elif k_list[count] == 'D':
			pyautogui.press('d')
		elif k_list[count] == 'F':
			pyautogui.press('f')
		count = count + 1
		if count == 18:
			break
"""
