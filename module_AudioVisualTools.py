# -*- coding: utf-8 -*-

"""
	MODULE_AUDIOVISUALTOOLS
	-----------------------

		This module implements several functions so it can be
		used as a toolbox for audiovisual problems.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- get_audioFrames
			
"""

from __future__ import division
import module_AudioTools as AT
import module_VideoTools as VT
import numpy as np
import os

def get_audioFrames(audioDir, videoDir, overlap, downsamplingFactor=1):
	"""
	FUNCTION NAME: get_audioFrames

	This function generates audio frames. In other words, 
	this function generates the audio files corresponding
	to the audio samples associated with a frame of video.
	Particularly, the generated audio frame overlaps with 
	the preceding and the consecutive audio frames.

	INPUT:
	------
		-> audioDir:			directory where the audio file is stored
		-> videoDir:			directory where the video file is stored
		-> overlap: 			percentage of overlap when generating 
								consecutive audio frames
		-> downsamplingFactor:	integer indicating the factor by which 
								to downsample the available data, if needed

	OUTPUT:
	-------

	"""

	fs, audioSamples = AT.readAudioFile(audioDir)
	frames, _, _, fps = VT.get_videoParameters(videoDir)

	# Check 0 <= overlap <= 1
	if overlap > 1:
		overlap = overlap/100.

	# Check directory to store the audioFrames
	if downsamplingFactor == 1:
		path = audioDir.split('.mp4.wav')[0]
		if (os.path.isdir(path) == False):
			os.mkdir(path)
	else:
		path = audioDir.split('.mp4.wav')[0] + '_downsampledDataBy' + str(downsamplingFactor)
		if (os.path.isdir(path) == False):
			os.mkdir(path)

	audioSamples_per_frame = int(fs*downsamplingFactor/fps)

	print '-> Analyzing file: ' + audioDir

	for frameCount in np.arange(int(np.floor(frames/downsamplingFactor))):

		print '	... Processing frame ' + str(int(frameCount+1)) + '/' + str(int(np.floor(frames/downsamplingFactor)))

		# Generate mask signal
		tmp = np.zeros(int(overlap*audioSamples_per_frame + audioSamples_per_frame + overlap*audioSamples_per_frame), dtype = np.int16)
		init_sample = int(frameCount * audioSamples_per_frame)

		if frameCount == 0:

			# main windowed signal
			tmp[int(overlap*audioSamples_per_frame):int(overlap*audioSamples_per_frame) + audioSamples_per_frame] = audioSamples[init_sample : init_sample + audioSamples_per_frame]
			
			# future overlapped signal
			tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame:] = audioSamples[init_sample + audioSamples_per_frame : init_sample + audioSamples_per_frame + int(overlap * audioSamples_per_frame)]


		elif frameCount == (int(np.floor(frames/downsamplingFactor)) - 1):
			
			# previous overlapped signal
			tmp[:int(overlap*audioSamples_per_frame)] = audioSamples[init_sample - int(overlap * audioSamples_per_frame) : init_sample]
			
			# main windowed signal
			tmp[int(overlap*audioSamples_per_frame):int(overlap*audioSamples_per_frame) + audioSamples_per_frame] = audioSamples[init_sample : init_sample + audioSamples_per_frame]
			
			# future overlapped signal
			if len(tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame:]) > len(audioSamples[init_sample + audioSamples_per_frame :]):
				tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame : int(overlap*audioSamples_per_frame) + audioSamples_per_frame + len(audioSamples[init_sample + audioSamples_per_frame :])] = audioSamples[init_sample + audioSamples_per_frame :]
			elif len(tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame:]) < len(audioSamples[init_sample + audioSamples_per_frame :]):	
				tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame : ] = audioSamples[init_sample + audioSamples_per_frame : init_sample + audioSamples_per_frame + len(tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame:])]

		else:
			# previous overlapped signal
			tmp[:int(overlap*audioSamples_per_frame)] = audioSamples[init_sample - int(overlap * audioSamples_per_frame): init_sample]
			
			# main windowed signal
			tmp[int(overlap*audioSamples_per_frame):int(overlap*audioSamples_per_frame) + audioSamples_per_frame] = audioSamples[init_sample : init_sample + audioSamples_per_frame]
			
			# future overlapped signal
			tmp[int(overlap*audioSamples_per_frame) + audioSamples_per_frame:] = audioSamples[init_sample + audioSamples_per_frame : init_sample + audioSamples_per_frame + int(overlap * audioSamples_per_frame)]

		# Write signal in directory
		AT.writeAudioFile(path + '/AudioFrame_%06d' % frameCount + '.wav', fs, tmp)

