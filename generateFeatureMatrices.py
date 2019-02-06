#!/usr/bin/python2
# Python 2 script

# Function call: python generateFeatureMatrices.py dataSplit featureType
#	-> dataSplit: 'training/validation'
#	-> featureType: 'audio/audio_subset/IS13/FAUs/audio+FAUs/audio_subset+FAUs/IS13+FAUs/visual/audiovisual'

import os
import sys
import module_OMGdata as OMG

if __name__ == '__main__':

	if (sys.argv[1] == 'training') or (sys.argv[1] == 'validation') or (sys.argv[1] == 'test'):
		if sys.argv[1] == 'training':
			# Train directory
			path = '/nas/staff/data_work/Adria/OMG_Challenge/Training'
		elif sys.argv[1] == 'validation':
			# Validation directory
			path = '/nas/staff/data_work/Adria/OMG_Challenge/Validation'
		elif sys.argv[1] == 'test':
			# Test directory
			path = '/nas/staff/data_work/Adria/OMG_Challenge/Test'

		if os.path.isdir(path + '/DataMatrices') == False:
			os.mkdir(path + '/DataMatrices')

		if sys.argv[2] == 'audio':
			
			if sys.argv[1] == 'test':
				OMG.build_audioFeatures_downsampledDataset_test(path, 5)
			else:
				OMG.build_audioFeatures_downsampledDataset(path, 5)
			
		elif sys.argv[2] == 'FAUs':
			
			if sys.argv[1] == 'test':
				OMG.build_FAUsFeatures_downsampledDataset_test(path, 5)
			else:
				OMG.build_FAUsFeatures_downsampledDataset(path, 5)

		elif sys.argv[2] == 'audio+FAUs':
			
			if sys.argv[1] == 'test':
				OMG.build_audioFAUsFeatures_downsampledDataset_test(path, 5)
			else:
				OMG.build_audioFAUsFeatures_downsampledDataset(path, 5)

	elif sys.argv[1] == 'training/validation':

		# Train directory
		path = '/nas/staff/data_work/Adria/OMG_Challenge/Training'

		if os.path.isdir(path + '/DataMatrices') == False:
			os.mkdir(path + '/DataMatrices')

		if sys.argv[2] == 'audio':
			
			OMG.build_audioFeatures_downsampledDataset(path, 5)

		elif sys.argv[2] == 'FAUs':
			
			OMG.build_FAUsFeatures_downsampledDataset(path, 5)

		elif sys.argv[2] == 'audio+FAUs':
			
			OMG.build_audioFAUsFeatures_downsampledDataset(path, 5)
	
		#############################################
		# Validation directory
		path = '/nas/staff/data_work/Adria/OMG_Challenge/Validation'

		if os.path.isdir(path + '/DataMatrices') == False:
			os.mkdir(path + '/DataMatrices')

		if sys.argv[2] == 'audio':
			
			OMG.build_audioFeatures_downsampledDataset(path, 5)
			
		elif sys.argv[2] == 'FAUs':
			
			OMG.build_FAUsFeatures_downsampledDataset(path, 5)

		elif sys.argv[2] == 'audio+FAUs':
			
			OMG.build_audioFAUsFeatures_downsampledDataset(path, 5)



