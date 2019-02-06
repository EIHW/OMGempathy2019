#!/usr/bin/python2
# Python 2 script

import os

# Path of videos 
video_folders = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/Videos/',
 	                '/nas/staff/data_work/Adria/OMG_Challenge/Validation/Videos/',
					'/nas/staff/data_work/Adria/OMG_Challenge/Test/Videos/']

output_folders = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/VideosListeners/',
	                 '/nas/staff/data_work/Adria/OMG_Challenge/Validation/VideosListeners/',
					'/nas/staff/data_work/Adria/OMG_Challenge/Test/VideosListeners/']

# Extract and convert visual features with openFACE
for folder_in, folder_out in zip(video_folders, output_folders):
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    
    for fn in os.listdir(folder_in):
        if fn.endswith('.mp4'):
        	os.system('ffmpeg -i ' + folder_in + fn + ' -vf "crop=iw/2:ih:iw/2:0" -an ' + folder_out + fn)
   
