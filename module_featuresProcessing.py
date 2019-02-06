# -*- coding: utf-8 -*-

"""
	MODULE_FEATURESPROCESSING
	-------------------------

		This module contains useful function regarding features processing.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- z_normalization
			- normalization_0_1
"""

import numpy as np

def z_normalization(features):
	"""
	FUNCTION NAME: z_normalization

	Function to z-normalize a matrix of features. .

	INPUT:
	------
		-> features:		matrix of features with dimensions (n_samples, n_features)

	OUTPUT:
	-------
		<- normFeatures:	matrix with normalized features, shape (n_samples, n_features)

	"""

	normFeatures = np.copy(features)

	for k in np.arange(np.shape(features)[1]):

		if np.std(features[:,k]) != 0:

			normFeatures[:,k] = np.divide(np.subtract(features[:,k], np.mean(features[:,k])), np.std(features[:,k]))

	return normFeatures

def normalization_0_1(features, minVal=None, maxVal=None):
	"""
	FUNCTION NAME: normalization_0_1

	Function to normalize the input features into a [0,1] range.
	If minVal and maxVal are None, the normalization will be 
	performed according to features maximum and minimum. 
	Otherwise, the normalization will be based on the minVal and
	maxVal values inputted to the function. This funcionality is
	especially designed for pre-ranged features that belong to
	a certain range different than [0,1].

	INPUT:
	------
		-> features:		features to normalize
		-> minVal:			minimum value of the pre-ranged
							features
		-> maxVal:			maximum value of the pre-ranged
							features

	OUTPUT:
	-------
		<- normFeat:		normalized features

	"""

	if (minVal == None) and (maxVal == None):

		normFeat = np.copy(features)

		for k in np.arange(np.shape(features)[1]):

			if np.subtract(np.max(features[:,k], axis=0), np.min(features[:,k], axis=0)) != 0:

				normFeat[:,k] = np.divide(np.subtract(features[:,k], np.min(features[:,k],axis=0)), \
									np.subtract(np.max(features[:,k],axis=0), np.min(features[:,k],axis=0)))
	else:

		normFeat = np.divide(np.subtract(features, minVal), np.subtract(maxVal, minVal))

	return normFeat