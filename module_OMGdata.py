# -*- coding: utf-8 -*-

"""
	MODULE_OMGDATA
	--------------

		This module implements several function to manage the
		data provided for the OMG-empathy challenge.

		IMPLEMENTED FUNCTIONS:
		----------------------
			- build_audioFeatures
			- build_audioFeatures_downsampledDataset
			- build_audioFeatures_downsampledDataset_test
			- build_FAUsFeatures
			- build_FAUsFeatures_test
			- build_FAUsFeatures_downsampledDataset
			- build_FAUsFeatures_downsampledDataset_test
			- build_audioFAUsFeatures
			- build_audioFAUsFeatures_downsampledDataset
			- build_audioFAUsFeatures_downsampledDataset_test
			- generate_PredictedAnnotationsFolder
			
"""

import os
import pandas as pd
import module_DataExtraction as DE
import module_VideoTools as VT
import module_DataRegeneration as DR

def build_audioFeatures(path):
	"""
	FUNCTION NAME: build_audioFeatures

	This function creates audio features and label csv files, so
	they can be used in the machine learning stage of our 
	working pipeline.

	INPUT:
	------
		-> path:		path to data

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'f_1', 'f_2', 'f_3','f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', \
					'f_11', 'f_12', 'f_13',	'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', \
					'f_21', 'f_22',	'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', \
					'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39',	'f_40', \
					'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', \
					'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56',	'f_57', 'f_58', 'f_59', 'f_60', \
					'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', \
					'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', \
					'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/openSMILE_features.csv'
	outputLabelsFile = path + '/DataMatrices/openSMILE_labels.csv'

	folders = sorted(os.listdir(path + '/AudioFeatures'))

	for folder in folders:

		if os.path.isdir(path + '/AudioFeatures' + '/' + folder):

			print 'Building data for: ' + folder + ' ...'

			files = sorted(os.listdir(path + '/AudioFeatures' + '/' + folder))

			for file in files:

				if file.endswith('.csv'):

					featuresFile = path + '/AudioFeatures/' + folder + '/' + file
					labelsFile = path + '/Annotations/' + folder + '.csv'

					frameNumber = int(file.split('.csv')[0][-6:])
					subjectID = int(folder.split('_')[1])
					storyID = int(folder.split('_')[3])

					# Read features information
					currentFeat = DE.read_openSMILEfeatures(featuresFile)
					# Reshape features information
					currentFeat = DE.reshape_openSMILEfeaturesVector(currentFeat)
					# Concatenate features information
					features = DE.concatenate_info(subjectID, storyID, frameNumber, currentFeat)
					
					# Read labels information
					annotation = DE.get_annotationFromFrameNumber(labelsFile, frameNumber)	
					# Reshape labels information
					annotation = DE.reshape_annotation(annotation)
					# Concatenate labels information
					labels = DE.concatenate_info(subjectID, storyID, frameNumber, annotation)
					
					# Write data in .csv file
					if header == True:
						f = pd.DataFrame(features)
						f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%e', index=False, mode='w')
						l = pd.DataFrame(labels)
						l.to_csv(outputLabelsFile, header=headerStringLabels, index=False, mode='w')
					else:
						f = pd.DataFrame(features)
						f.to_csv(outputFeaturesFile, header=header, float_format='%e', index=False, mode='a')
						l = pd.DataFrame(labels)
						l.to_csv(outputLabelsFile, header=header, index=False, mode='a')

					header = False

def build_audioFeatures_downsampledDataset(path, downsamplingFactor):
	"""
	FUNCTION NAME: build_audioFeatures_downsampledDataset

	This function creates audio features and label csv files, so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used when the 
	dataset available is reduced by a certain factor.

	INPUT:
	------
		-> path:					path to data
		-> downsamplingFactor:		factor by which the original dataset 
									is downsampled

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'f_1', 'f_2', 'f_3','f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', \
					'f_11', 'f_12', 'f_13',	'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', \
					'f_21', 'f_22',	'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', \
					'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39',	'f_40', \
					'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', \
					'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56',	'f_57', 'f_58', 'f_59', 'f_60', \
					'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', \
					'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', \
					'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/openSMILE_downsamplingFactor' + str(downsamplingFactor) + '_features.csv'
	outputLabelsFile = path + '/DataMatrices/openSMILE_downsamplingFactor' + str(downsamplingFactor) + '_labels.csv'

	folders = sorted(os.listdir(path + '/AudioFeatures'))

	for folder in folders:

		if (os.path.isdir(path + '/AudioFeatures' + '/' + folder)) and (len(folder.split('_')) == 5) and \
				(folder.split('_')[4] == 'downsampledDataBy' + str(downsamplingFactor)):

			print 'Building data for: ' + folder + ' ...'

			files = sorted(os.listdir(path + '/AudioFeatures' + '/' + folder))

			for file in files:

				if file.endswith('.csv'):

					featuresFile = path + '/AudioFeatures/' + folder + '/' + file
					labelsFile = path + '/Annotations/' + folder.split('_downsampledDataBy')[0] + '.csv'

					frameNumber = int(file.split('.csv')[0][-6:])
					subjectID = int(folder.split('_')[1])
					storyID = int(folder.split('_')[3])

					# Read features information
					currentFeat = DE.read_openSMILEfeatures(featuresFile)
					# Reshape features information
					currentFeat = DE.reshape_openSMILEfeaturesVector(currentFeat)
					# Concatenate features information
					features = DE.concatenate_info(subjectID, storyID, int(downsamplingFactor*frameNumber), currentFeat)
					
					# Read labels information
					annotation = DE.get_annotationFromFrameNumber(labelsFile, int(downsamplingFactor*frameNumber))	
					# Reshape labels information
					annotation = DE.reshape_annotation(annotation)
					# Concatenate labels information
					labels = DE.concatenate_info(subjectID, storyID, int(downsamplingFactor*frameNumber), annotation)
					
					# Write data in .csv file
					if header == True:
						f = pd.DataFrame(features)
						f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%e', index=False, mode='w')
						l = pd.DataFrame(labels)
						l.to_csv(outputLabelsFile, header=headerStringLabels, index=False, mode='w')
					else:
						f = pd.DataFrame(features)
						f.to_csv(outputFeaturesFile, header=header, float_format='%e', index=False, mode='a')
						l = pd.DataFrame(labels)
						l.to_csv(outputLabelsFile, header=header, index=False, mode='a')

					header = False

def build_audioFeatures_downsampledDataset_test(path, downsamplingFactor):
	"""
	FUNCTION NAME: build_audioFeatures_downsampledDataset_test

	This function creates audio features as csv files, so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used when the 
	dataset available is reduced by a certain factor and when 
	preparing the features to be used during test.

	INPUT:
	------
		-> path:					path to data
		-> downsamplingFactor:		factor by which the original dataset 
									is downsampled

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'f_1', 'f_2', 'f_3','f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', \
					'f_11', 'f_12', 'f_13',	'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', \
					'f_21', 'f_22',	'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', \
					'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39',	'f_40', \
					'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', \
					'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56',	'f_57', 'f_58', 'f_59', 'f_60', \
					'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', \
					'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', \
					'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/openSMILE_downsamplingFactor' + str(downsamplingFactor) + '_features.csv'

	folders = sorted(os.listdir(path + '/AudioFeatures'))

	for folder in folders:

		if (os.path.isdir(path + '/AudioFeatures' + '/' + folder)) and (len(folder.split('_')) == 5) and \
				(folder.split('_')[4] == 'downsampledDataBy' + str(downsamplingFactor)):

			print 'Building data for: ' + folder + ' ...'

			files = sorted(os.listdir(path + '/AudioFeatures' + '/' + folder))

			for file in files:

				if file.endswith('.csv'):

					featuresFile = path + '/AudioFeatures/' + folder + '/' + file
					
					frameNumber = int(file.split('.csv')[0][-6:])
					subjectID = int(folder.split('_')[1])
					storyID = int(folder.split('_')[3])

					# Read features information
					currentFeat = DE.read_openSMILEfeatures(featuresFile)
					# Reshape features information
					currentFeat = DE.reshape_openSMILEfeaturesVector(currentFeat)
					# Concatenate features information
					features = DE.concatenate_info(subjectID, storyID, int(downsamplingFactor*frameNumber), currentFeat)
					
					# Write data in .csv file
					if header == True:
						f = pd.DataFrame(features)
						f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%e', index=False, mode='w')
					else:
						f = pd.DataFrame(features)
						f.to_csv(outputFeaturesFile, header=header, float_format='%e', index=False, mode='a')
						
					header = False

def build_FAUsFeatures(path):
	"""
	FUNCTION NAME: build_FAUsFeatures

	This function creates FAUs features and label .csv files so
	they can be used in the machine learning stage of our 
	working pipeline.

	INPUT:
	------
		-> path:		path to data

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/FAUs_features.csv'
	outputLabelsFile = path + '/DataMatrices/FAUs_labels.csv'

	files = sorted(os.listdir(path + '/VideoFeatures'))

	for file in files:

		if file.endswith('.csv'):

			print 'Building data for: ' + file[:-4] + ' ...'

			featuresFile = path + '/VideoFeatures/' + file
			labelsFile = path + '/Annotations/' + file

			subjectID = int(file[:-4].split('_')[1])
			storyID = int(file[:-4].split('_')[3])

			# Read FAUs information
			currentFAUs = DE.read_FAUs(featuresFile)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			# Concatenate FAUs information
			features = DE.concatenate_info_FAUs(subjectID, storyID, framesID, currentFAUs)
			
			# Read labels information
			annotations = DE.get_annotationsFromFile(labelsFile)

			# Concatenate labels information
			labels = DE.concatenate_info_FAUs(subjectID, storyID, framesID, annotations)
			
			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%.2f', index=False, mode='w')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=headerStringLabels, index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%.2f', index=False, mode='a')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=header, index=False, mode='a')

			header = False

def build_FAUsFeatures_test(path):
	"""
	FUNCTION NAME: build_FAUsFeatures_test

	This function creates FAUs features .csv files so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used
	when generating features data from the testing set,
	without labels.

	INPUT:
	------
		-> path:		path to data

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']
	
	outputFeaturesFile = path + '/DataMatrices/FAUs_features.csv'
	
	files = sorted(os.listdir(path + '/VideoFeatures'))

	for file in files:

		if file.endswith('.csv'):

			print 'Building data for: ' + file[:-4] + ' ...'

			featuresFile = path + '/VideoFeatures/' + file
	
			subjectID = int(file[:-4].split('_')[1])
			storyID = int(file[:-4].split('_')[3])

			# Read FAUs information
			currentFAUs = DE.read_FAUs(featuresFile)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			# Concatenate FAUs information
			features = DE.concatenate_info_FAUs(subjectID, storyID, framesID, currentFAUs)
			
			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%.2f', index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%.2f', index=False, mode='a')
			
			header = False

def build_FAUsFeatures_downsampledDataset(path, downsamplingFactor):
	"""
	FUNCTION NAME: build_FAUsFeatures_downsampledDataset

	This function creates FAUs features and label .csv files so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used when the 
	dataset available is reduced by a certain factor.

	INPUT:
	------
		-> path:					path to data
		-> downsamplingFactor:		factor by which the original dataset 
									is downsampled

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/FAUs_downsamplingFactor' + str(downsamplingFactor) + '_features.csv'
	outputLabelsFile = path + '/DataMatrices/FAUs_downsamplingFactor' + str(downsamplingFactor) + '_labels.csv'

	files = sorted(os.listdir(path + '/VideoFeatures'))

	for file in files:

		if file.endswith('.csv'):

			print 'Building data for: ' + file[:-4] + ' ...'

			featuresFile = path + '/VideoFeatures/' + file
			labelsFile = path + '/Annotations/' + file

			subjectID = int(file[:-4].split('_')[1])
			storyID = int(file[:-4].split('_')[3])

			# Read FAUs information
			currentFAUs = DE.read_FAUs(featuresFile)

			# Select the proper instances according to the downsampling
			dwIDs = DE.get_downsamplingIDs(currentFAUs, downsamplingFactor)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			# Concatenate FAUs information
			features = DE.concatenate_info_FAUs(subjectID[dwIDs,:], storyID[dwIDs,:], framesID[dwIDs,:], currentFAUs[dwIDs,:])
			
			# Read labels information
			annotations = DE.get_annotationsFromFile(labelsFile)

			# Concatenate labels information
			labels = DE.concatenate_info_FAUs(subjectID[dwIDs,:], storyID[dwIDs,:], framesID[dwIDs,:], annotations[dwIDs,:])
			
			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%.2f', index=False, mode='w')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=headerStringLabels, index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%.2f', index=False, mode='a')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=header, index=False, mode='a')

			header = False

def build_FAUsFeatures_downsampledDataset_test(path, downsamplingFactor):
	"""
	FUNCTION NAME: build_FAUsFeatures_downsampledDataset

	This function creates FAUs features .csv files so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used when the 
	dataset available is reduced by a certain factor.
	This function is aimed to be used when generating features data
	from the testing set.

	INPUT:
	------
		-> path:					path to data
		-> downsamplingFactor:		factor by which the original dataset 
									is downsampled

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']

	outputFeaturesFile = path + '/DataMatrices/FAUs_downsamplingFactor' + str(downsamplingFactor) + '_features.csv'

	files = sorted(os.listdir(path + '/VideoFeatures'))

	for file in files:

		if file.endswith('.csv'):

			print 'Building data for: ' + file[:-4] + ' ...'

			featuresFile = path + '/VideoFeatures/' + file
			
			subjectID = int(file[:-4].split('_')[1])
			storyID = int(file[:-4].split('_')[3])

			# Read FAUs information
			currentFAUs = DE.read_FAUs(featuresFile)

			# Select the proper instances according to the downsampling
			dwIDs = DE.get_downsamplingIDs(currentFAUs, downsamplingFactor)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			# Concatenate FAUs information
			features = DE.concatenate_info_FAUs(subjectID[dwIDs,:], storyID[dwIDs,:], framesID[dwIDs,:], currentFAUs[dwIDs,:])
			
			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%.2f', index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%.2f', index=False, mode='a')
			
			header = False

def build_audioFAUsFeatures(path):
	"""
	FUNCTION NAME: build_audioFAUsFeatures

	This function creates openSMILE and FAUs features and 
	label .csv files so	they can be used in the machine 
	learning stage of our working pipeline.

	INPUT:
	------
		-> path:		path to data

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'f_1', 'f_2', 'f_3','f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', \
					'f_11', 'f_12', 'f_13',	'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', \
					'f_21', 'f_22',	'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', \
					'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39',	'f_40', \
					'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', \
					'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56',	'f_57', 'f_58', 'f_59', 'f_60', \
					'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', \
					'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', \
					'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/openSMILE_FAUs_features.csv'
	outputLabelsFile = path + '/DataMatrices/openSMILE_FAUs_labels.csv'

	folders = sorted(os.listdir(path + '/AudioFeatures'))

	for folder in folders:

		if os.path.isdir(path + '/AudioFeatures' + '/' + folder):

			print 'Building data for: ' + folder + ' ...'

			files = sorted(os.listdir(path + '/AudioFeatures' + '/' + folder))

			# FAU features path
			FAUpath = path + '/VideoFeatures/' + folder + '.csv'
			# Annotations path
			valencePath = path + '/Annotations/' + folder + '.csv'

			subjectID = int(folder.split('_')[1])
			storyID = int(folder.split('_')[3])

			print '	Reading openSMILE features information ...'

			openSMILEfeatures = []

			for file in files:

				if file.endswith('.csv'):

					openSMILEpath = path + '/AudioFeatures/' + folder + '/' + file

					# Read features information
					currentFeat = DE.read_openSMILEfeatures(openSMILEpath)
					# Reshape features information
					currentFeat = DE.reshape_openSMILEfeaturesVector(currentFeat)

					# Concatenate openSMILE features information
					if len(openSMILEfeatures) == 0:
						openSMILEfeatures = currentFeat
					else:
						openSMILEfeatures = DE.incrementalMatrix(openSMILEfeatures, currentFeat)

			print '	Reading FAUs features information ...'

			# Read FAUs information
			currentFAUs = DE.read_FAUs(FAUpath)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			print '	Reading valence labels information ...'

			# Read valence labels information
			annotations = DE.get_annotationsFromFile(valencePath)

			# Concatenate labels information
			labels = DE.concatenate_info_FAUs(subjectID, storyID, framesID, annotations)

			# Concatenate openSMILE + FAUs features information
			features = DE.concatenate_info_openSMILE_FAUs(subjectID, storyID, framesID, openSMILEfeatures, currentFAUs)

			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%e', index=False, mode='w')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=headerStringLabels, index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%e', index=False, mode='a')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=header, index=False, mode='a')

			header = False

def build_audioFAUsFeatures_downsampledDataset(path, downsamplingFactor):
	"""
	FUNCTION NAME: build_audioFAUsFeatures_downsampledDataset

	This function creates audio+FAUs features and label .csv files so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used when the 
	dataset available is reduced by a certain factor. 

	INPUT:
	------
		-> path:					path to data
		-> downsamplingFactor:		factor by which the original dataset 
									is downsampled

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'f_1', 'f_2', 'f_3','f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', \
					'f_11', 'f_12', 'f_13',	'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', \
					'f_21', 'f_22',	'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', \
					'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39',	'f_40', \
					'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', \
					'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56',	'f_57', 'f_58', 'f_59', 'f_60', \
					'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', \
					'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', \
					'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']
	headerStringLabels = ['subjectID', 'storyID', 'frameID', 'valence']

	outputFeaturesFile = path + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(downsamplingFactor) + '_features.csv'
	outputLabelsFile = path + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(downsamplingFactor) + '_labels.csv'

	folders = sorted(os.listdir(path + '/AudioFeatures'))

	for folder in folders:

		if (os.path.isdir(path + '/AudioFeatures' + '/' + folder)) and (len(folder.split('_')) == 5) and \
				(folder.split('_')[4] == 'downsampledDataBy' + str(downsamplingFactor)):

			print 'Building data for: ' + folder + ' ...'

			files = sorted(os.listdir(path + '/AudioFeatures' + '/' + folder))

			# FAU features path
			FAUpath = path + '/VideoFeatures/' + folder.split('_downsampledDataBy')[0] + '.csv'
			# Annotations path
			valencePath = path + '/Annotations/' + folder.split('_downsampledDataBy')[0] + '.csv'

			subjectID = int(folder.split('_')[1])
			storyID = int(folder.split('_')[3])

			print '	Reading openSMILE features information ...'

			openSMILEfeatures = []

			for file in files:

				if file.endswith('.csv'):

					openSMILEpath = path + '/AudioFeatures/' + folder + '/' + file

					# Read features information
					currentFeat = DE.read_openSMILEfeatures(openSMILEpath)
					# Reshape features information
					currentFeat = DE.reshape_openSMILEfeaturesVector(currentFeat)

					# Concatenate openSMILE features information
					if len(openSMILEfeatures) == 0:
						openSMILEfeatures = currentFeat
					else:
						openSMILEfeatures = DE.incrementalMatrix(openSMILEfeatures, currentFeat)

			print '	Reading FAUs features information ...'

			# Read FAUs information
			currentFAUs = DE.read_FAUs(FAUpath)

			# Select the proper instances according to the downsampling
			dwIDs = DE.get_downsamplingIDs(currentFAUs, downsamplingFactor)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			print '	Reading valence labels information ...'

			# Read valence labels information
			annotations = DE.get_annotationsFromFile(valencePath)

			# Concatenate labels information
			labels = DE.concatenate_info_FAUs(subjectID[dwIDs,:], storyID[dwIDs,:], framesID[dwIDs,:], annotations[dwIDs,:])

			# Concatenate openSMILE + FAUs features information
			features = DE.concatenate_info_openSMILE_FAUs(subjectID[dwIDs,:], storyID[dwIDs,:], framesID[dwIDs,:], openSMILEfeatures, currentFAUs[dwIDs,:])

			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%e', index=False, mode='w')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=headerStringLabels, index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%e', index=False, mode='a')
				l = pd.DataFrame(labels)
				l.to_csv(outputLabelsFile, header=header, index=False, mode='a')

			header = False

def build_audioFAUsFeatures_downsampledDataset_test(path, downsamplingFactor):
	"""
	FUNCTION NAME: build_audioFAUsFeatures_downsampledDataset

	This function creates audio+FAUs features .csv files so
	they can be used in the machine learning stage of our 
	working pipeline. This function is aimed to be used when the 
	dataset available is reduced by a certain factor. This function
	is aimed to be used when generating features data from the 
	testing set.

	INPUT:
	------
		-> path:					path to data
		-> downsamplingFactor:		factor by which the original dataset 
									is downsampled

	OUTPUT:
	-------

	"""

	features = []
	labels = []
	header = True
	headerStringFeatures = ['subjectID', 'storyID', 'frameID', \
					'f_1', 'f_2', 'f_3','f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', \
					'f_11', 'f_12', 'f_13',	'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', \
					'f_21', 'f_22',	'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', \
					'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39',	'f_40', \
					'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', \
					'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56',	'f_57', 'f_58', 'f_59', 'f_60', \
					'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', \
					'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', \
					'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88', \
					'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', \
					'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
					'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', \
					'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', \
					'AU26_c', 'AU28_c', 'AU45_c']
	
	outputFeaturesFile = path + '/DataMatrices/openSMILE_FAUs_downsamplingFactor' + str(downsamplingFactor) + '_features.csv'
	
	folders = sorted(os.listdir(path + '/AudioFeatures'))

	for folder in folders:

		if (os.path.isdir(path + '/AudioFeatures' + '/' + folder)) and (len(folder.split('_')) == 5) and \
				(folder.split('_')[4] == 'downsampledDataBy' + str(downsamplingFactor)):

			print 'Building data for: ' + folder + ' ...'

			files = sorted(os.listdir(path + '/AudioFeatures' + '/' + folder))

			# FAU features path
			FAUpath = path + '/VideoFeatures/' + folder.split('_downsampledDataBy')[0] + '.csv'
			
			subjectID = int(folder.split('_')[1])
			storyID = int(folder.split('_')[3])

			print '	Reading openSMILE features information ...'

			openSMILEfeatures = []

			for file in files:

				if file.endswith('.csv'):

					openSMILEpath = path + '/AudioFeatures/' + folder + '/' + file

					# Read features information
					currentFeat = DE.read_openSMILEfeatures(openSMILEpath)
					# Reshape features information
					currentFeat = DE.reshape_openSMILEfeaturesVector(currentFeat)

					# Concatenate openSMILE features information
					if len(openSMILEfeatures) == 0:
						openSMILEfeatures = currentFeat
					else:
						openSMILEfeatures = DE.incrementalMatrix(openSMILEfeatures, currentFeat)

			print '	Reading FAUs features information ...'

			# Read FAUs information
			currentFAUs = DE.read_FAUs(FAUpath)

			# Select the proper instances according to the downsampling
			dwIDs = DE.get_downsamplingIDs(currentFAUs, downsamplingFactor)

			subjectID, storyID, framesID = DE.get_infoFromFAUs(currentFAUs, subjectID, storyID)

			# Concatenate openSMILE + FAUs features information
			features = DE.concatenate_info_openSMILE_FAUs(subjectID[dwIDs,:], storyID[dwIDs,:], framesID[dwIDs,:], openSMILEfeatures, currentFAUs[dwIDs,:])

			# Write data in .csv file
			if header == True:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=headerStringFeatures, float_format='%e', index=False, mode='w')
			else:
				f = pd.DataFrame(features)
				f.to_csv(outputFeaturesFile, header=header, float_format='%e', index=False, mode='a')
			
			header = False

def generate_PredictedAnnotationsFolder(validationPath, folderPath, dw):
	"""
	FUNCTION NAME: generate_PredictedAnnotationsFolder

	Function to organize the predicted labels according to subject and
	stories IDs.

	INPUT:
	------
		-> validationPath:	path of validation data
		-> folderPath:		folder path where to store the data, which
							corresponds to the name of the predicted
							annotations files without the extension
		-> dw:				integer indicating the factor by which the available
							data is downsampled
		
	OUTPUT:
	-------

	"""

	headerLabel = ['valence']

	results = pd.DataFrame(pd.read_csv(folderPath + '.csv')).values

	subjectIDs = DR.get_uniqueValues(results[:,0])
	storyIDs = DR.get_uniqueValues(results[:,1])

	for subject in subjectIDs:
		
		for story in storyIDs:

			outputFile = 'Subject_' + str(int(subject)) + '_Story_' + str(int(story))
			
			print '-> Processing ' + outputFile + ' file ...'

			numOfFrames, _, _, _ = VT.get_videoParameters(validationPath + '/Videos/' + outputFile + '.mp4')

			currentArguments = DR.getArguments_SubjectID_StoryID(results, subject, story)

			if (dw != 1):
				labels = DR.regenerateAnnotations(results[currentArguments,:], int(numOfFrames))
			else:
				labels = results[currentArguments, 3]

			l = pd.DataFrame(labels)
			l.to_csv(folderPath + '/' + outputFile + '.csv', header=headerLabel, index=False, mode='w')

			print '<- Processed!'
