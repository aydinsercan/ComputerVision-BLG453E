import numpy as np
import os
import cv2
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import glob

main_dir = "C:/Users/SERCAN/Desktop/homework1/cat"
all_images = os.listdir(main_dir)

target_dir = "C:/Users/SERCAN/Desktop/homework1/"
target_img = cv2.imread(target_dir)

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
	
	gamma = 0.2; #gamma 1'den büyük olursa brightness, tam tersi darkness
	gamma_corrected = 255*(image/255)**(1/gamma)
	
	nonzero_cat_values = image[nonzero_x, nonzero_y,:]
	new_frame = background.copy()
	new_frame[nonzero_x,nonzero_y, :] = nonzero_cat_values #left side
	nonzero_cat_values = np.uint8(gamma_corrected[nonzero_x, nonzero_y,:])
	new_frame[nonzero_x,-nonzero_y, :] = nonzero_cat_values #rigth side
	new_frame = new_frame[:,:,[2,1,0]] 
	frame_list.append(new_frame)

clip = mpy.ImageSequenceClip(frame_list, fps=25)
audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video.mp4', codec = 'libx264')
