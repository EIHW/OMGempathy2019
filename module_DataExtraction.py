# -*- coding: utf-8 -*-

"""
	MODULE_DATAEXTRACTION
	---------------------

		This module implements several functions so it can be
		used as a toolbox regarding feature extraction.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- read_openSMILEfeatures
			- read_openSMILE_IS13features
			- concatenate_openSMILE_IS13features_headerInfo
			- reshape_openSMILEfeaturesVector
			- read_FAUs
			- get_infoFromFAUs
			- get_downsamplingIDs
			- get_annotationsFromFile
			- get_annotationFromFrameNumber
			- reshape_annotation
			- incrementalMatrix
			- concatenate_info
			- concatenate_info_FAUs
			- concatenate_info_openSMILE_FAUs
			- get_uniqueValues
			- getArguments_SubjectID_StoryID
			- getArguments_SubjectID
			- tensorCreation
			- get_3DtensorDimensions
			- BLSTM_reshape
			- dataMatrices2tensors
			
"""

import numpy as np
import pandas as pd

def read_openSMILEfeatures(filePath):
	"""
	FUNCTION NAME: read_openSMILEfeatures

	This function reads a .csv file containing openSMILE
	features and builds an array with the corresponding 
	information.

	INPUT:
	------
		-> filePath:	path of the features .csv file
						to read

	OUTPUT:
	-------
		<- features:	array with the features read from
						the input file

	"""

	features = np.genfromtxt(filePath,delimiter=';',dtype=float)[1:]

	return features

def read_openSMILE_IS13features(filePath):
	"""
	FUNCTION NAME: read_openSMILE_IS13features

	This function reads a .csv file containing openSMILE
	features and builds an array with the corresponding 
	information.

	INPUT:
	------
		-> filePath:	path of the features .csv file
						to read

	OUTPUT:
	-------
		<- features:	array with the features read from
						the input file

	"""

	features = np.genfromtxt(filePath,delimiter=';',skip_header=1,dtype=float)[1:]

	return features

def read_openSMILE_IS13features_header(filePath):
	"""
	FUNCTION NAME: read_openSMILE_IS13features_header

	This function reads a .csv file containing openSMILE
	features and returns the headers of the columns to extract.

	INPUT:
	------
		-> filePath:	path of the features .csv file
						to read

	OUTPUT:
	-------
		<- header:		array of strings

	"""

	header = np.genfromtxt(filePath,delimiter=';',max_rows=1,dtype=str)[1:]

	return header

def concatenate_openSMILE_IS13features_headerInfo(featHeader, info):
	"""
	FUNCTION NAME: concatenate_openSMILE_IS13features_headerInfo

	Function to concatente openSMILE features information and 
	information when writting a global file.

	INPUT:
	------
		-> featHeader:		array of string with features information
		-> info:			array of strings with additional information

	OUTPUT:
	-------
		<- header:			concatenation of information with the appropriate
							shape

	"""

	header = np.hstack((info, featHeader))
	header = np.reshape(header,(1,-1))

	return header

def reshape_openSMILEfeaturesVector(inVec):
	"""
	FUNCTION NAME: reshape_openSMILEfeaturesVector

	Function to reshape openSMILE feature vector so it can be
	concatenated with other ones into a matrix that can be fed
	into the machine learning stage of a working pipeline.

	INPUT:
	------
		-> inVec:	original vector extracted from read_openSMILEfeatures
					function

	OUTPUT:
	-------
		<- outVec:	reshaped openSMILE feature vector with size 1x88

	"""

	outVec = np.reshape(inVec,(1,-1))

	return outVec

def read_FAUs(filePath):
	"""
	FUNCTION NAME: read_FAUs

	This function reads a .csv file containing FAUs extracted using
	openFace and builds an array with the corresponding information.

	INPUT:
	------
		-> filePath: 	path of the FAUs .csv file to read

	OUTPUT:
	-------
		<- FAUs:		array with the FAUs read from the input 
						file

	"""

	obj = pd.read_csv(filePath)
	FAUs_pd = obj.drop(obj.iloc[:,0:141], axis=1)
	df = pd.DataFrame(FAUs_pd)
	FAUs = df.values

	return FAUs

def get_infoFromFAUs(FAUs, subjectID, storyID):
	"""
	FUNCTION NAME: get_infoFromFAUs

	Function to compute an array with frames IDs included in a 
	FAUs feature vector. In addition, the function also returns
	a subjectID and a story ID vector with the same length as 
	the number of frames.

	INPUT:
	------
		-> FAUs:			completed FAUs features matrix
		-> subjectID:		subject ID to replicate
		-> storyID: 		storyID to replicate

	OUTPUT:
	-------
		<- subjectVecID:	array with subject IDs
		<- storyVecID:		array with story IDs
		<- framesID: 		array with frames IDs

	"""

	framesID = np.arange(np.shape(FAUs)[0])
	framesID = np.reshape(framesID,(-1,1))
	subjectVecID = subjectID * np.ones((np.shape(framesID)))
	storyVecID = storyID * np.ones((np.shape(framesID)))

	return subjectVecID, storyVecID, framesID

def get_downsamplingIDs(features, downsamplingFactor):
	"""
	FUNCTION NAME: get_downsamplingIDs

	Function to return the IDs from the original features matrix
	to be used when the dataset is reduced by a certain downsampling
	factor.

	INPUT:
	------
		-> features:				matrix of features from the original
									dataset
		-> downsamplingFactor:		factor by which to reduce the original 
									number of instances

	OUTPUT:
	-------
		<- IDs:						array with the IDs to select from the 
									original features

	"""

	IDs = np.arange(0,np.shape(features)[0],downsamplingFactor)

	return IDs

def get_annotationsFromFile(annotationFile):
	"""
	FUNCTION NAME: get_annotationFromFile

	Function to extract the whole annotations from the input file
	and return them as a numpy array.

	INPUT:
	------
		-> annotationFile:		path where the annotations are stored

	OUTPUT:
	-------
		<- annotations:			array with the annotations to retrieve

	"""

	annotations = np.genfromtxt(annotationFile, dtype=float)[1:]
	annotations = np.reshape(annotations,(-1,1))

	return annotations

def get_annotationFromFrameNumber(annotationFile, frameNum):
	"""
	FUNCTION NAME: 	get_annotationFromFrameNumber

	This function returns the annotation corresponding to a 
	particular frame stored in the annotation file of the dataset.

	INPUT:
	------
		-> annotationFile:		path where the annotations are stored
		-> frameNum:			number corresponding to the frame to 
								retrieve the annotation from

	OUTPUT:
	-------
		<- annotation:			annotation to retrieve

	"""

	fileContent = np.genfromtxt(annotationFile, dtype=float)[1:]

	annotation = fileContent[int(frameNum)]

	return annotation

def reshape_annotation(oldVal):
	"""
	FUNCTION NAME: reshape_annotation

	Function to reshape the annotation value so it matches
	the data structure to be fed into the machine learning
	stage of the pipeline.

	INPUT:
	------
		-> oldVal: 		annotation value stored in file

	OUTPUT:
	-------
		<- newVal:		annotation value reshaped

	"""

	newVal = np.reshape(oldVal, (1,1))

	return newVal

def incrementalMatrix(oldMatrix, newArray):
	"""
	FUNCTION NAME: incrementalMatrix

	Function to stack a new array into an already existing
	matrix. This function can be used when incrementally 
	building features matrices from disk files.

	INPUT:
	------
		-> oldMatrix:		already existing matrix
		-> newArray:		new array to stack to the already
							existing matrix

	OUTPUT:
	-------
		<- newMatrix:		matrix obtained from the concatenation

	"""

	newMatrix = np.vstack((oldMatrix, newArray))

	return newMatrix

def concatenate_info(subjectID, storyID, frameID, data):
	"""
	FUNCTION NAME: concatenate_info

	Function to concatenate the input parameters in a row vector.

	INPUT:
	------
		-> subjectID:	ID corresponding to the subject
		-> storyID:		ID corresponding to the story
		-> frameID:		ID corresponding to the frame
		-> data:		data to be concatenated with

	OUTPUT:
	-------
		<- oVec:		vector with the information concatenated

	"""

	oVec = np.hstack((np.reshape([subjectID, storyID, frameID],(1,-1)),data))

	return oVec

def concatenate_info_FAUs(subjectID, storyID, framesID, data):
	"""
	FUNCTION NAME: concatenate_info_FAUs

	This function concatenates subjectID, storyID and frameID arrays
	when input data are FAUs.

	INPUT:
	------
		-> subjectID:	ID corresponding to the subject
		-> storyID:		ID corresponding to the story
		-> framesID:	ID corresponding to the frame
		-> data:		data to be concatenated with

	OUTPUT:
	-------
		<- oVec:		vector with the information concatenated

	"""

	oVec = np.hstack((np.hstack((np.hstack((subjectID, storyID)), framesID)),data))

	return oVec

def concatenate_info_openSMILE_FAUs(subjectID, storyID, framesID, data1, data2):
	"""
	FUNCTION NAME: concatenate_info_openSMILE_FAUs

	This function concatenates subjectID, storyID and frameID arrays
	when input data are openSMILE and FAUs features.

	INPUT:
	------
		-> subjectID:	ID corresponding to the subject
		-> storyID:		ID corresponding to the story
		-> framesID:	ID corresponding to the frame
		-> data1:		openSMILE features to concatenate
		-> data2: 		FAUs features to concatenate

	OUTPUT:
	-------
		<- oVec:		vector with the information concatenated

	"""

	oVec = np.hstack((np.hstack((np.hstack((np.hstack((subjectID, storyID)), framesID)),data1)), data2))

	return oVec

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

def getArguments_SubjectID(outputAnnotations, subjectID):
	"""
	FUNCTION NAME: getArguments_SubjectID

	This function returns the rows of the outputAnnotations matrix 
	that contain information from a especific subject.

	INPUT:
	------
		-> outputAnnotations:		data matrix computed when testing
									the empathy models trained
		-> subjectID:				ID of the subject to retrieve the 
									information from

	OUTPUT:
	-------
		<- rowArg:					arguments of the rows corresponding
									to the specified subject

	"""

	rowArg = np.argwhere(outputAnnotations[:,0] == subjectID)

	return rowArg

def tensorCreation(num_samples, num_timesteps, num_features):
	"""
	FUNCTION NAME: tensorCreation

	This function creates a tensor with the dimensions specified
	by the input parameters.

	INPUT:
	------
		-> num_samples:		first dimension of the tensor
		-> num_timesteps:	second dimension of the tensor
		-> num_features:	third dimension of the tensor

	OUTPUT:
	-------
		<- tensor:			zero tensor with the desired shape

	"""

	tensor = np.zeros((num_samples, num_timesteps, num_features))

	return tensor

def get_3DtensorDimensions(tensor):
	"""
	FUNCTION NAME: get_3DtensorDimensions

	Function to return the shape of the input 3D tensor.

	INPUT:
	------
		-> tensor:		input tensor to compute the shape

	OUTPUT:
	-------
		<- d1:			1st dimension of the tensor
		<- d2:			2nd dimension of the tensor
		<- d3:			3rd dimension of the tensor
	
	"""

	[d1, d2, d3] = np.shape(tensor)

	return d1, d2, d3

def BLSTM_reshape(features, batch_size):
	"""
	FUNCTION NAME: BLSTM_reshape

	Function to reshape the features matrix to fit the proper data
	format to be fed into a Bidirectional LSTM model.

	INPUT:
	------
		-> features:		original feature matrix to reshape
		-> batch_size:		first dimension of the reshaped matrix

	OUTPUT:
	-------	
		<- reshaped:		reshaped feature matrix

	"""

	reshaped = np.reshape(features,(batch_size, np.shape(features)[0], np.shape(features)[1]))

	return reshaped

def dataMatrices2tensors(features, labels, modelType):
	"""
	FUNCTION NAME: dataMatrices2tensors

	This function converts the input data matrices into tensors. 
	The dimensions of the output tensor dimensions correspond to 
	[# subjects, # frames, # features].

	INPUT:
	------
		-> features:		features matrix with data [subjectID, storyID, frameID, features]
		-> labels:			labels matrix with data [subjectID, storyID, frameID, labels]
		-> modelType:		string indicating the nature of the input features

	OUTPUT:
	-------
		<- T_features:		features tensor
		<- T_labels:		labels tensor

	"""

	if modelType == 'openSMILE':
		n_features = 88
	elif modelType == 'FAUs':
		n_features = 35
	elif modelType == 'openSMILE+FAUs':
		n_features = 88+35
	elif modelType == 'openSMILE_subset':
		n_features = 46
	elif modelType == 'openSMILE_subset+FAUs':
		n_features = 81
	elif modelType == 'openSMILE_IS13':
		n_features = 6373
	elif modelType == 'openSMILE_IS13+FAUs':
		n_features = 6373+35

	subjectIDs = get_uniqueValues(features[:,0])
	storyIDs = get_uniqueValues(features[:,1])

	nSamples = 0
	maxLen = 0

	for sbID in subjectIDs:
		for stID in storyIDs:

			nSamples += 1
			current_rows = getArguments_SubjectID_StoryID(features, sbID, stID)

			if len(current_rows) > maxLen:
				maxLen = len(current_rows)

	T_features = tensorCreation(nSamples, maxLen, n_features)
	T_labels = tensorCreation(nSamples, maxLen, 1)

	nSamples = 0
	for sbID in subjectIDs:
		for stID in storyIDs:

			current_rows = getArguments_SubjectID_StoryID(features, sbID, stID)

			T_features[nSamples,:len(current_rows),:] = features[current_rows,3:]
			T_labels[nSamples,:len(current_rows),:] = labels[current_rows,3:]			

			nSamples += 1

	return T_features, T_labels
