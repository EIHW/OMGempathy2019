# -*- coding: utf-8 -*-

"""
	MODULE_MACHINELEARNING_PERSONALIZEDTRACK
	----------------------------------------

		This module implements several functions so it can be
		used as a toolbox when working on machine learning 
		stages of our pipelines.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- loss_CCC
			- train_model_BLSTM_variableSequenceLength
			- test_model_BLSTM_variableSequenceLength
			- train_model_2BLSTM_variableSequenceLength
			- test_model_2BLSTM_variableSequenceLength
			- train_model
			- validate_model
			- train_TrainDev_model
			- test_TrainDev_model

"""

import os
import numpy as np
import pickle
import pandas as pd
import module_featuresProcessing as FP
import module_DataExtraction as DE
import module_OMGdata as OMG
import tensorflow as tf
from sklearn.svm import SVR
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Masking
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from scipy.signal import medfilt

def loss_CCC(seq1, seq2):
	"""
	FUNCTION NAME: loss_CCC

	This function implements the Concordance Correlation Coefficient (CCC)
	to be used as a loss function to train models.

	INPUT:
	------
		-> seq1:		tensor with the true output: (num_batches, seq_len, 1)
		-> seq2:		tensor with the predicted output: (num_batches, seq_len, 1)

	OUTPUT:
	-------
		<- cccLoss:		(1 - CCC) computed to be used as a CCC loss

	"""

	seq1 = K.squeeze(seq1, axis=-1)
	seq2 = K.squeeze(seq2, axis=-1)
	seq1_mean = K.mean(seq1, axis=-1, keepdims=True)
	seq2_mean = K.mean(seq2, axis=-1, keepdims=True)
	cov = (seq1-seq1_mean)*(seq2-seq2_mean)
	seq1_var = K.mean(K.square(seq1-seq1_mean), axis=-1, keepdims=True)
	seq2_var = K.mean(K.square(seq2-seq2_mean), axis=-1, keepdims=True)

	CCC = K.constant(2.) * cov / (seq1_var + seq2_var + K.square(seq1_mean - seq2_mean) + K.epsilon())
	CCC_loss = K.constant(1.) - CCC

	return CCC_loss

def train_model_BLSTM_variableSequenceLength(path, subjectID, modelType, MLtechnique, features, labels, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: train_model_BLSTM_variableSequenceLength

	This function trains a bidirectional LSTM model when input sequences
	have difference length from one sample to the other. In the first step,
	input sequences are arranged into a tensor of the same length using
	zero padding. When data is ready, the bidirectional LSTM is trained.

	INPUT:
	------
		-> path:			full path where to store the trained model
		-> subjectID:		integer indicating the ID of the subject being analyzed
		-> modelType:		type of model to train
		-> MLtechnique:		technique to use to train the model
		-> features:		matrix of features to train the model
		-> labels:			matrix of labels to train the model
		-> dw:				factor used when downsampling the available data
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of units of the LSTM
		
	OUTPUT:
	------- 

	"""

	epochs = 200
	verbose = 1

	if (dw == 1):
		modelName = path + 'Model_Subject' + str(subjectID) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType
	else:
		modelName = path + 'Model_Subject' + str(subjectID) + '_DW' + str(dw) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType

	# Convert data matrices to tensors
	T_features, T_labels = DE.dataMatrices2tensors(features, labels, modelType)

	# Define the Bidirectional LSTM
	model = Sequential([
				Masking(mask_value = 0., input_shape=(None,DE.get_3DtensorDimensions(T_features)[2])),
				Bidirectional(LSTM(LSTMunits, activation='tanh', return_sequences=True)),
				TimeDistributed(Dense(1, activation='linear'))
				])

	model.compile(optimizer=Adam(),loss=loss_CCC)

	earlyStop = EarlyStopping(monitor='loss', patience=patience)
	callbacks_list = [earlyStop]

	# Train the model
	model.fit(T_features, T_labels, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks = callbacks_list, validation_split = 0)

	print '-> Saving model ..'
	# Save model
	model.save(modelName + '.h5')
	print '<- Model saved'	

def test_model_BLSTM_variableSequenceLength(inputPath, outputPath, modelType, MLtechnique, features, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: test_model_BLSTM_variableSequenceLength

	Function to test B-LSTM models trained on the testing/validation sets.

	INPUT:
	------
		-> inputPath:		path where trained models are stored
		-> outputPath:		path where the validation data needs to be stored
		-> modelType:		type of model to train
		-> MLtechnique:		technique to use to train the model
		-> features:		matrix of features to validate the model
		-> dw:				factor used when downsampling the available data
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of units of the LSTM

	OUTPUT:
	-------
		<- predictions:		numpy array with the annotations predicted from
							the input features with data structure:
								[subjectID, storyID, frameID, predictions]

	"""

	predictions = []

	subjectIDs = DE.get_uniqueValues(features[:,0])
	storyIDs = DE.get_uniqueValues(features[:,1])

	for sbID in subjectIDs:

		if (dw == 1):
			modelName = inputPath + 'Model_Subject' + str(int(sbID)) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType
		else:
			modelName = inputPath + 'Model_Subject' + str(int(sbID)) + '_DW' + str(dw) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType

		print '-> Loading model from disk ...'
		# Load model
		model = load_model(modelName + '.h5', custom_objects={'loss_CCC': loss_CCC})
		print '<- Model loaded!'

		for stID in storyIDs:

			current_rows = DE.getArguments_SubjectID_StoryID(features, sbID, stID)

			selectedFeatures = features[current_rows, 3:]
			modelInputFeatures = np.reshape(selectedFeatures,(1,np.shape(selectedFeatures)[0], np.shape(selectedFeatures)[1]))
			pred_annotations = np.squeeze(model.predict(modelInputFeatures))

			# Apply median filtering to the valence predictions
			pred_annotations = medfilt(pred_annotations, 301)

			currentOutput = np.hstack((features[current_rows,:3], np.reshape(pred_annotations,(-1,1))))

			if len(predictions) == 0:
				predictions = currentOutput
			else:
				predictions = DE.incrementalMatrix(predictions, currentOutput)

		del model

	return predictions

def train_model_2BLSTM_variableSequenceLength(path, subjectID, modelType, MLtechnique, features, labels, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: train_model_2BLSTM_variableSequenceLength

	This function trains a bidirectional LSTM model with 2 hidden layers
	when input sequences have difference length from one sample to the other. 
	In the first step, input sequences are arranged into a tensor of the same 
	length using zero padding. When data is ready, the bidirectional LSTM is trained.

	INPUT:
	------
		-> path:			full path where to store the trained model
		-> subjectID:		integer indicating the ID of the subject being analyzed
		-> modelType:		type of model to train
		-> MLtechnique:		technique to use to train the model
		-> features:		matrix of features to train the model
		-> labels:			matrix of labels to train the model
		-> dw:				factor used when downsampling the available data
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of units of the LSTM
		
	OUTPUT:
	------- 

	"""

	epochs = 200
	verbose = 1

	if (dw == 1):
		modelName = path + 'Model_Subject' + str(subjectID) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType
	else:
		modelName = path + 'Model_Subject' + str(subjectID) + '_DW' + str(dw) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType

	# Convert data matrices to tensors
	T_features, T_labels = DE.dataMatrices2tensors(features, labels, modelType)

	# Define the Bidirectional LSTM
	model = Sequential([
				Masking(mask_value = 0., input_shape=(None,DE.get_3DtensorDimensions(T_features)[2])),
				Bidirectional(LSTM(LSTMunits, activation='tanh', return_sequences=True)),
				Bidirectional(LSTM(int(LSTMunits/2), activation='tanh', return_sequences=True)),
				TimeDistributed(Dense(1, activation='linear'))
				])

	model.compile(optimizer=Adam(),loss=loss_CCC)

	earlyStop = EarlyStopping(monitor='loss', patience=patience)
	callbacks_list = [earlyStop]

	# Train the model
	model.fit(T_features, T_labels, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks = callbacks_list, validation_split = 0)

	print '-> Saving model ..'
	# Save model
	model.save(modelName + '.h5')
	print '<- Model saved'	

def test_model_2BLSTM_variableSequenceLength(inputPath, outputPath, modelType, MLtechnique, features, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: test_model_2BLSTM_variableSequenceLength

	Function to test 2B-LSTM models trained on the testing/validation sets.

	INPUT:
	------
		-> inputPath:		path where trained models are stored
		-> outputPath:		path where the validation data needs to be stored
		-> modelType:		type of model to train
		-> MLtechnique:		technique to use to train the model
		-> features:		matrix of features to validate the model
		-> dw:				factor used when downsampling the available data
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of units of the LSTM

	OUTPUT:
	-------
		<- predictions:		numpy array with the annotations predicted from
							the input features with data structure:
								[subjectID, storyID, frameID, predictions]

	"""

	predictions = []

	subjectIDs = DE.get_uniqueValues(features[:,0])
	storyIDs = DE.get_uniqueValues(features[:,1])

	for sbID in subjectIDs:

		if (dw == 1):
			modelName = inputPath + 'Model_Subject' + str(int(sbID)) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType
		else:
			modelName = inputPath + 'Model_Subject' + str(int(sbID)) + '_DW' + str(dw) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType

		print '-> Loading model from disk ...'
		# Load model
		model = load_model(modelName + '.h5', custom_objects={'loss_CCC': loss_CCC})
		print '<- Model loaded!'

		for stID in storyIDs:

			current_rows = DE.getArguments_SubjectID_StoryID(features, sbID, stID)

			selectedFeatures = features[current_rows, 3:]
			modelInputFeatures = np.reshape(selectedFeatures,(1,np.shape(selectedFeatures)[0], np.shape(selectedFeatures)[1]))
			pred_annotations = np.squeeze(model.predict(modelInputFeatures))

			# Apply median filtering to the valence predictions
			pred_annotations = medfilt(pred_annotations, 301)

			currentOutput = np.hstack((features[current_rows,:3], np.reshape(pred_annotations,(-1,1))))

			if len(predictions) == 0:
				predictions = currentOutput
			else:
				predictions = DE.incrementalMatrix(predictions, currentOutput)

		del model

	return predictions

def train_model(dataPath, modelType, MLtechnique, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: train_model

	Function to train a desired model with an specific machine
	learning technique.

	INPUT:
	------
		-> datPath:			full path where data is stored
		-> modelType:		type of model to train
		-> MLtechnique:		technique to use to train the model
		-> dw:				integer indicating the factor by which the available
							data is downsampled
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of LSTM cells in case B-LSTM is used

	OUTPUT:
	-------

	"""

	features = []

	if not os.path.isdir(dataPath + '/Models_PersonalizedTrack'):
		os.mkdir(dataPath + '/Models_PersonalizedTrack')

	if (MLtechnique == 'B-LSTM'):

		if (modelType == 'openSMILE'):
		
				if (dw == 1):
					# Load features
					features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_features.csv')).values
					# Load labels
					labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_labels.csv')).values
				else:
					# Load features
					features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
					# Load labels
					labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_labels.csv')).values
				
				# z-normalize openSMILE features
				numOfOpenSMILEfeatures = 88
				features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif (modelType == 'FAUs'):
			if (dw == 1):
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_labels.csv')).values
			else:
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values

		elif (modelType == 'openSMILE+FAUs'):
			if (dw == 1):
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_labels.csv')).values
			else:
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values
			
			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])
		
		subjectIDs = DE.get_uniqueValues(features[:,0])
		for sbID in subjectIDs:
			current_rows = np.squeeze(DE.getArguments_SubjectID(features, sbID))
			train_model_BLSTM_variableSequenceLength(dataPath + '/Models_PersonalizedTrack/', int(sbID), modelType, MLtechnique, features[current_rows,:], labels[current_rows,:], dw, batch_size, patience, LSTMunits=LSTMunits)	

	elif (MLtechnique == '2B-LSTM'):

		if (modelType == 'openSMILE'):
		
				if (dw == 1):
					# Load features
					features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_features.csv')).values
					# Load labels
					labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_labels.csv')).values
				else:
					# Load features
					features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
					# Load labels
					labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_labels.csv')).values
				
				# z-normalize openSMILE features
				numOfOpenSMILEfeatures = 88
				features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif (modelType == 'FAUs'):
			if (dw == 1):
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_labels.csv')).values
			else:
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values

		elif (modelType == 'openSMILE+FAUs'):
			if (dw == 1):
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_labels.csv')).values
			else:
				# Load features
				features = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				# Load labels
				labels = pd.DataFrame(pd.read_csv(dataPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values
			
			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])
		
		subjectIDs = DE.get_uniqueValues(features[:,0])
		for sbID in subjectIDs:
			current_rows = np.squeeze(DE.getArguments_SubjectID(features, sbID))
			train_model_2BLSTM_variableSequenceLength(dataPath + '/Models_PersonalizedTrack/', int(sbID), modelType, MLtechnique, features[current_rows,:], labels[current_rows,:], dw, batch_size, patience, LSTMunits=LSTMunits)

def validate_model(trainingPath, validationPath, modelType, MLtechnique, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: validate_model

	Function to test trained models on the validation set.

	INPUT:
	------
		-> trainingPath:	path where training data is stored
		-> validationPath:	path where validation data is stored
		-> modelType:		string indicating the type of models to validate
		-> MLtechnique:		string indicating the type of ML technique to use
		-> dw:				integer indicating the factor by which the available
							data is downsampled
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of LSTM cells in case B-LSTM is used

	OUTPUT:
	-------

	"""

	if not os.path.isdir(validationPath + '/Models_Output_PersonalizedTrack'):
		os.mkdir(validationPath + '/Models_Output_PersonalizedTrack')

	if MLtechnique == 'B-LSTM':

		if (modelType == 'openSMILE'):

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
			
			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif (modelType == 'FAUs'):

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
		
		elif (modelType == 'openSMILE+FAUs'):

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		prediction = test_model_BLSTM_variableSequenceLength(trainingPath + '/Models_PersonalizedTrack/', validationPath + '/Models_Output/', modelType, MLtechnique, features, dw, batch_size, patience, LSTMunits=LSTMunits)

	elif MLtechnique == '2B-LSTM':

		if (modelType == 'openSMILE'):

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
			
			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif (modelType == 'FAUs'):

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
		
		elif (modelType == 'openSMILE+FAUs'):

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		prediction = test_model_2BLSTM_variableSequenceLength(trainingPath + '/Models_PersonalizedTrack/', validationPath + '/Models_Output/', modelType, MLtechnique, features, dw, batch_size, patience, LSTMunits=LSTMunits)

	# Write results in file
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']
	outputFile = validationPath + '/Models_Output_PersonalizedTrack/' + 'PredictedValence_DW' + str(dw) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType + '.csv'
	l = pd.DataFrame(prediction)
	l.to_csv(outputFile, header=headerStringLabels, index=False, mode='w')

	# Post-process output data to match the required format
	if os.path.isfile(outputFile):
		if not os.path.isdir(outputFile.split('.csv')[0]):
			os.mkdir(outputFile.split('.csv')[0])

		OMG.generate_PredictedAnnotationsFolder(validationPath, outputFile.split('.csv')[0], dw)

def train_TrainDev_model(trainingPath, validationPath, modelType, MLtechnique, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION NAME: train_TrainDev_model

	Function to train a model using data from training and validation (or development)
	sets. 

	INPUT:
	------
		-> trainingPath:	path where training data is stored
		-> validationPath:	path where validation data is stored
		-> modelType:		string indicating the type of models to validate
		-> MLtechnique:		string indicating the type of ML technique to use
		-> dw:				integer indicating the factor by which the available
							data is downsampled
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of LSTM cells in case B-LSTM is used

	OUTPUT:
	-------

	"""

	if not os.path.isdir(trainingPath + '/TrainDevModels_PersonalizedTrack'):
		os.mkdir(trainingPath + '/TrainDevModels_PersonalizedTrack')

	if MLtechnique == 'B-LSTM':

		if modelType == 'openSMILE':

			if (dw == 1):
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_labels.csv')).values
			else:
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_labels.csv')).values

			features = np.vstack((featuresT, featuresV))
			labels = np.vstack((labelsT, labelsV))

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif modelType == 'FAUs':

			if (dw == 1):
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_labels.csv')).values
			else:
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values

			features = np.vstack((featuresT, featuresV))
			labels = np.vstack((labelsT, labelsV))

		elif modelType == 'openSMILE+FAUs':

			if (dw == 1):
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_labels.csv')).values
			else:
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values

			features = np.vstack((featuresT, featuresV))
			labels = np.vstack((labelsT, labelsV))

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		subjectIDs = DE.get_uniqueValues(features[:,0])
		for sbID in subjectIDs:
			current_rows = np.squeeze(DE.getArguments_SubjectID(features, sbID))
			train_model_BLSTM_variableSequenceLength(trainingPath + '/TrainDevModels_PersonalizedTrack/', int(sbID), modelType, MLtechnique, features[current_rows,:], labels[current_rows,:], dw, batch_size, patience, LSTMunits=LSTMunits)	

	elif MLtechnique == '2B-LSTM':

		if modelType == 'openSMILE':

			if (dw == 1):
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_labels.csv')).values
			else:
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_labels.csv')).values

			features = np.vstack((featuresT, featuresV))
			labels = np.vstack((labelsT, labelsV))

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif modelType == 'FAUs':

			if (dw == 1):
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_labels.csv')).values
			else:
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values

			features = np.vstack((featuresT, featuresV))
			labels = np.vstack((labelsT, labelsV))

		elif modelType == 'openSMILE+FAUs':

			if (dw == 1):
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_labels.csv')).values
			else:
				# Load features and labels training set
				featuresT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsT = pd.DataFrame(pd.read_csv(trainingPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values
				# Load features and labels validation set
				featuresV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values
				labelsV = pd.DataFrame(pd.read_csv(validationPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_labels.csv')).values

			features = np.vstack((featuresT, featuresV))
			labels = np.vstack((labelsT, labelsV))

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		subjectIDs = DE.get_uniqueValues(features[:,0])
		for sbID in subjectIDs:
			current_rows = np.squeeze(DE.getArguments_SubjectID(features, sbID))
			train_model_2BLSTM_variableSequenceLength(trainingPath + '/TrainDevModels_PersonalizedTrack/', int(sbID), modelType, MLtechnique, features[current_rows,:], labels[current_rows,:], dw, batch_size, patience, LSTMunits=LSTMunits)	

def test_TrainDev_model(trainingPath, testPath, modelType, MLtechnique, dw, batch_size, patience, LSTMunits=30):
	"""
	FUNCTION  NAME: test_TrainDev_model

	Function to test the models trained with Train+Dev data splits.

	INPUT:
	------
		-> trainingPath:	path where training data is stored
		-> testPath:		path where test data is stored
		-> modelType:		string indicating the type of models to validate
		-> MLtechnique:		string indicating the type of ML technique to use
		-> dw:				integer indicating the factor by which the available
							data is downsampled
		-> batch_size:		value for batch_size parameter
		-> patience:		value for patience parameter
		-> LSTMunits:		number of LSTM cells in case B-LSTM is used

	OUTPUT:
	-------

	"""

	if not os.path.isdir(testPath + '/TrainDevModels_PersonalizedTrack_Output'):
		os.mkdir(testPath + '/TrainDevModels_PersonalizedTrack_Output')

	if MLtechnique == 'B-LSTM':

		if modelType == 'openSMILE':

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
			
			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif modelType == 'FAUs':

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values

		elif modelType == 'openSMILE+FAUs':

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		prediction = test_model_BLSTM_variableSequenceLength(trainingPath + '/TrainDevModels_PersonalizedTrack/', testPath + '/TrainDevModels_PersonalizedTrack_Output/', modelType, MLtechnique, features, dw, batch_size, patience, LSTMunits=LSTMunits)

	elif MLtechnique == '2B-LSTM':

		if modelType == 'openSMILE':

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_downsamplingFactor' + str(dw) + '_features.csv')).values
			
			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		elif modelType == 'FAUs':

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values

		elif modelType == 'openSMILE+FAUs':

			# Load features
			if (dw == 1):
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_FAUs_features.csv')).values
			else:
				features = pd.DataFrame(pd.read_csv(testPath + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(dw) + '_features.csv')).values

			# z-normalize openSMILE features
			numOfOpenSMILEfeatures = 88
			features[:,3:3+numOfOpenSMILEfeatures] = FP.z_normalization(features[:,3:3+numOfOpenSMILEfeatures])

		prediction = test_model_2BLSTM_variableSequenceLength(trainingPath + '/TrainDevModels_PersonalizedTrack/', testPath + '/TrainDevModels_PersonalizedTrack_Output/', modelType, MLtechnique, features, dw, batch_size, patience, LSTMunits=LSTMunits)

	# Write results in file
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']
	outputFile = testPath + '/TrainDevModels_PersonalizedTrack_Output/' + 'PredictedValence_DW' + str(dw) + '_' + MLtechnique + '_LSTMunits' + str(LSTMunits) + '_BatchSize' + str(batch_size) + '_Patience' + str(patience) + '_' + modelType + '.csv'

	l = pd.DataFrame(prediction)
	l.to_csv(outputFile, header=headerStringLabels, index=False, mode='w')

	# Post-process output data to match the required format
	if os.path.isfile(outputFile):
		if not os.path.isdir(outputFile.split('.csv')[0]):
			os.mkdir(outputFile.split('.csv')[0])

		OMG.generate_PredictedAnnotationsFolder(testPath, outputFile.split('.csv')[0], dw)
