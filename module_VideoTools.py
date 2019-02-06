# -*- coding: utf-8 -*-

"""
	MODULE_VIDEOTOOLS
	-----------------

		This module implements several functions so it can be
		used as a toolbox for video problems.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- get_videoParameters
			
"""

import cv2

def get_videoParameters(fileIn):
	"""
	FUNCTION NAME: get_videoParameters

	This function returns length, width, height and fps of the
	input video.

	INPUT:
	------
		-> fileIn: 		path corresponding to the video to extract
						the sampling rate from

	OUTPUT:
	-------
		<- length:		total number of video frames
		<- width:		width of video frames
		<- height: 		height of video frames
		<- fps:			frames per second of the input video

	"""

	vidObject = cv2.VideoCapture(fileIn)
	
	length = vidObject.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	width = vidObject.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	height = vidObject.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	fps = vidObject.get(cv2.cv.CV_CAP_PROP_FPS)

	return length, width, height, fps

