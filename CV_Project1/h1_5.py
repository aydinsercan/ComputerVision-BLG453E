import numpy as np
import os
import cv2
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import glob

main_dir = "C:/Users/SERCAN/Desktop/homework1/cat"
all_images = os.listdir(main_dir)


def noise(img):
	gauss = np.random.normal(0,1,img.size)
	gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
	# Add the Gaussian noise to the image
	img_gauss = cv2.add(img,gauss)
	return img_gauss


def get_LUT(cdf_img, cdf_target):
    m = cdf_img.shape[0]
    LUT = np.zeros(m)
    g_t = 0
    for g_i in range(m):
        while cdf_target[g_t] < cdf_img[g_i] and g_t < 255 :
            g_t += 1
        LUT[g_i] = g_t
    return LUT


def get_hist(patch, dim):
    hist = np.zeros((256, 1))
    for j in range(256):
        hist[j] = np.count_nonzero(patch[:, :, dim] == j)
    return hist


def hist_match(image, target):
    for i in range(3):
        hist_image = get_hist(image, i)
        #calculate histogram of the ith channel for the image
        cdf_image = hist_image.cumsum() / hist_image.sum()
        hist_target = get_hist(target, i)
        #calculate histogram of the ith channel for the target image
        cdf_target = hist_target.cumsum() / hist_target.sum()
        LUT_i = get_LUT(cdf_image, cdf_target)
        image[:,:,i] = np.uint8(LUT_i[image[:,:,i]])
        #finally, map each point according to the Look-Up Table
    return image


background = cv2.imread('Malibu.jpg')
background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360/background_height
background = cv2.resize(background,(int(background_width*ratio),360))

frame_list = []

data_path = os.path.join(main_dir,'*.png') 
files = sorted(glob.glob(data_path), key=os.path.getmtime) 

for f1 in files:
	image = cv2.imread(f1)
	image_g_channel = image[:,:,1]
	image_r_channel = image[:,:,0]
	foreground = np.logical_or(image_g_channel<180, image_r_channel>150)
	nonzero_x , nonzero_y = np.nonzero(foreground)
	
	perturb1 = noise(image)
	image1 = hist_match(image, perturb1)
	nonzero_cat_values = image1[nonzero_x, nonzero_y,:]
	new_frame = background.copy()
	new_frame[nonzero_x,nonzero_y, :] = nonzero_cat_values #Left place
	
	perturb2 = noise(image)
	image2 = hist_match(image, perturb2)
	nonzero_cat_values = image2[nonzero_x, nonzero_y,:]
	new_frame[nonzero_x,-nonzero_y, :] = nonzero_cat_values #right side
	
	new_frame = new_frame[:,:,[2,1,0]] 
	frame_list.append(new_frame)


clip = mpy.ImageSequenceClip(frame_list, fps=25)
audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video.mp4', codec = 'libx264')


