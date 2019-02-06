#!/usr/bin/python2
# Python 2 script

# FUNCTION CALL: 
# --------------
# $ python main_EmpathyPrediction.py -action arg1 -model arg2 -ml arg3 [-dw arg4 -LSTMunits arg5 -bs arg6 -p arg7]
#
#	-> arg1: train/validate/sanityCheck/train+validate/test/...
# 	-> arg2: all/openSMILE/FAUs/openSMILE+FAUs
#	-> arg3: B-LSTM/2B-LSTM...
#	-> arg4: 1,5
#	-> arg5: 30,40,50,60,...
#	-> arg6: 1,5,10,...
#	-> arg7: 2,3,...
#
#	[ ] indicates optional arguments

import os
import sys
import module_MachineLearning as ML
import module_OMGdata as OMG

if __name__ == '__main__':

	dw = 1
	unit = 30
	batch_size = 1
	patience = 2

	for idx, iArg in enumerate(sys.argv[1:], start = 1):
		if (idx + 1) < len(sys.argv):

			if iArg == '-action':
				action = sys.argv[idx + 1]

			elif iArg == '-model':
				model = sys.argv[idx + 1]

			elif iArg == '-ml':
				MLtechnique = sys.argv[idx + 1]

			elif iArg == '-dw':
				dw = int(sys.argv[idx + 1])

			elif iArg == '-bs':
				batch_size = int(sys.argv[idx + 1])

			elif iArg == '-p':
				patience = int(sys.argv[idx + 1])

			elif iArg == '-LSTMunits':
				unit = int(sys.argv[idx + 1])

	if action == 'gridSearch':
		trainingPath = '/nas/staff/data_work/Adria/OMG_Challenge/Training'
		validationPath = '/nas/staff/data_work/Adria/OMG_Challenge/Validation'
		ML.gridSearch(trainingPath, validationPath, model, MLtechnique, dw)

	elif action == 'GS_PostProcessing':
		validationPath = '/nas/staff/data_work/Adria/OMG_Challenge/Validation'
		OMG.gridSearch_PostProcessing(validationPath, model, MLtechnique, dw)
	
	elif action == 'train':
		dataPath = '/nas/staff/data_work/Adria/OMG_Challenge/Training'

		if model == 'all':
			ML.train_model(dataPath, 'openSMILE', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.train_model(dataPath, 'FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.train_model(dataPath, 'openSMILE+FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
		else:
			ML.train_model(dataPath, model, MLtechnique, dw, batch_size, patience, LSTMunits = unit)

	elif action == 'validate':
		trainingPath = '/nas/staff/data_work/Adria/OMG_Challenge/Training'
		validationPath = '/nas/staff/data_work/Adria/OMG_Challenge/Validation'

		if model == 'all':
			ML.validate_model(trainingPath, validationPath, 'openSMILE', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.validate_model(trainingPath, validationPath, 'FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.validate_model(trainingPath, validationPath, 'openSMILE+FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)

		else:
			ML.validate_model(trainingPath, validationPath, model, MLtechnique, dw, batch_size, patience, LSTMunits = unit)

	elif action == 'sanityCheck':
		trainingPath = '/nas/staff/data_work/Adria/OMG_Challenge/Training'

		if model == 'all':
			ML.validate_model(trainingPath, trainingPath, 'openSMILE', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.validate_model(trainingPath, trainingPath, 'FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.validate_model(trainingPath, trainingPath, 'openSMILE+FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)

		else:
			ML.validate_model(trainingPath, trainingPath, model, MLtechnique, dw, batch_size, patience, LSTMunits = unit)

	elif action == 'train+validate':
		trainingPath = '/nas/staff/data_work/Adria/OMG_Challenge/Training'
		validationPath = '/nas/staff/data_work/Adria/OMG_Challenge/Validation'

		if model == 'all':
			ML.train_TrainDev_model(trainingPath, validationPath, 'openSMILE', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.train_TrainDev_model(trainingPath, validationPath, 'FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.train_TrainDev_model(trainingPath, validationPath, 'openSMILE+FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)

		else:
			ML.train_TrainDev_model(trainingPath, validationPath, model, MLtechnique, dw, batch_size, patience, LSTMunits = unit)
		
	elif action == 'test':
		trainingPath = '/nas/staff/data_work/Adria/OMG_Challenge/Training'
		testPath = '/nas/staff/data_work/Adria/OMG_Challenge/Test'

		if model == 'all':
			ML.test_TrainDev_model(trainingPath, testPath, 'openSMILE', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.test_TrainDev_model(trainingPath, testPath, 'FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)
			ML.test_TrainDev_model(trainingPath, testPath, 'openSMILE+FAUs', MLtechnique, dw, batch_size, patience, LSTMunits = unit)

		else:
			ML.test_TrainDev_model(trainingPath, testPath, model, MLtechnique, dw, batch_size, patience, LSTMunits = unit)
