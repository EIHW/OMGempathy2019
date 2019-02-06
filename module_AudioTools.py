# -*- coding: utf-8 -*-

"""
	MODULE_AUDIOTOOLS
	-----------------

		This module implements several functions so it can be
		used as a toolbox for audio problems.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- readAudioFile
			- writeAudioFile
			- get_utteranceLength
			- checkChannels
"""

from scipy.io import wavfile
import numpy as np

def readAudioFile(audioFileName):
	"""
	FUNCTION NAME: readAudioFile

	This function reads the input audio file and returns and array with
	its corresponding samples:

	INPUT:
	------
		-> audioFileName: 		path of the audio file to read, including file name

	OUTPUT:
	-------
		<- samplingRate:		sampling rate of the audio file
		<- data:				array with the samples corresponding to the input audio file

	"""
	samplingRate, data = wavfile.read(audioFileName, 'r')

	return samplingRate, data

def writeAudioFile(audioFileName, samplingRate, data):
	"""
	FUNCTION NAME: writeAudioFile

	This function generates a .wav audio file from the input data at the specified
	sampling rate.

	INPUT:
	------
		-> audioFileName:		path of the audio to write, including file name
		-> samplingRate:		sampling rate of the audio file to generates
		-> data:				array with the information of the audio file to generate

	OUTPUT:
	-------

	"""

	wavfile.write(audioFileName, samplingRate, data)

def get_utteranceLength(audioFileName):
	"""
	FUNCTION NAME: get_utteranceLength

	This function retrieves the sampling rate of the input
	utterance and its length in samples.

	INPUT:
	------
		-> audioFileName: 		full path to load the audio file to analyze

	OUTPUT:
	-------
		<- samplingRate:		sampling rate in which the utterance was recorded
		<- utteranceLength:		length of the utterance in samples

	"""

	samplingRate, data = wavfile.read(audioFileName, 'r')
	utteranceLength = len(data)

	return samplingRate, utteranceLength

def checkChannels(InData):
	"""
	FUNCTION NAME: checkChannels

	This function checks the channels of the input data signal to ensure
	that all channels are non-zeros.

	INPUT:
	------
		-> InData: 		input signal

	OUTPUT:
	-------
		<- OutData:		output signal

	"""

	channelsSum = np.sum(InData,axis=0)

	if all([c != 0 for c in channelsSum]):
		OutData = InData
	else:
		OutData = InData[:, np.argwhere(channelsSum)[0][0]]
		OutData = np.reshape(OutData,(-1,1))

	return OutData