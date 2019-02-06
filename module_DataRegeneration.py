# -*- coding: utf-8 -*-

"""
	MODULE_DATAREGENERATION
	-----------------------

		This module implements several functions so it can be
		used as a toolbox regarding annotations data regeneration

		IMPLEMENTED FUNCTIONS:
		----------------------
			- get_uniqueValues
			- getArguments_SubjectID_StoryID
			- regenerateAnnotations

"""

import numpy as np
import pandas as pd

def get_uniqueValues(vector):
	"""
	FUNCTION NAME: get_uniqueValues

	This function returns the unique values in the input
	vector sorted.

	INPUT:
	------
		-> vector:		input vector to compute the unique values
						from

	OUTPUT:
	-------
		<- oVec:		output vector with the unique values of the 
						input vector

	"""

	oVec = sorted(np.unique(vector))

	return oVec

def getArguments_SubjectID_StoryID(outputAnnotations, subjectID, storyID):
	"""
	FUNCTION NAME: getArguments_SubjectID_StoryID

	This function returns the rows of the outputAnnotations matrix 
	that contain information from a especific subject and story.

	INPUT:
	------
		-> outputAnnotations:		data matrix computed when testing
									the empathy models trained
		-> subjectID:				ID of the subject to retrieve the 
									information from
		-> storyID:					ID of the story to retrieve the 
									information from

	OUTPUT:
	-------
		<- rowArg:					arguments of the rows corresponding
									to the specified subject and story

	"""

	rowArg = np.intersect1d(np.argwhere(outputAnnotations[:,0] == subjectID), np.argwhere(outputAnnotations[:,1] == storyID))

	return rowArg

def regenerateAnnotations(results, numOfFrames):
	"""
	FUNCTION NAME: regenerateAnnotations

	Function to upsample the predicted annotations to match
	with the original annotations using replication

	INPUT:
	-----
		-> results:					data to upsample. Its columns follow the
									structure [subjectID, storyID, frameID, annotations]
		-> numOfFrames:				total number of annotations to produce
		
	OUTPUT:
	-------
		<- annotations:				upsampled version of the predicted
									annotations

	"""

	annotations = np.zeros((numOfFrames,))

	dataCounter = 0

	for k in np.arange(0,numOfFrames):

		annotations[k] = results[dataCounter, 3]
		
		if (k < numOfFrames) and ((dataCounter + 1) < np.shape(results)[0]) and ((k + 1) == results[dataCounter + 1, 2]):
			dataCounter += 1

	return annotations
