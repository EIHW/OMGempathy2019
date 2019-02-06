#!/usr/bin/python2
# Python 2 script

import os
import module_AudioVisualTools as AVT

# Audio directory
audio_directory = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/Audio/',
	                '/nas/staff/data_work/Adria/OMG_Challenge/Validation/Audio/',
					'/nas/staff/data_work/Adria/OMG_Challenge/Test/Audio/']

# Video directory
video_directory = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/Videos/',
	                '/nas/staff/data_work/Adria/OMG_Challenge/Validation/Videos/',
					'/nas/staff/data_work/Adria/OMG_Challenge/Test/Videos/']

for audio_folder, video_folder in zip(audio_directory, video_directory):

	for file in os.listdir(video_folder):

		if file.endswith('.mp4'):

			audioPath = audio_folder + file + '.wav'
			videoPath = video_folder + file
			#AVT.get_audioFrames(audioPath, videoPath, .5)
			AVT.get_audioFrames(audioPath, videoPath, .5, 5)

