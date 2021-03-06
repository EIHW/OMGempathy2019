# EIHW system for OMG Empathy 2019 Challenge

This repository contains our approach's implementation for the OMG Empathy Prediction Challenge (https://www2.informatik.uni-hamburg.de/wtm/omgchallenges/omg_empathy.html). Below, we detail the execution order of the different scripts, define their call, and give a brief overview on what do they do. Please note that data paths in the main scripts should be adapted to each machine.

The following scripts are responsibles for processing the videos, extract features, and generate feature files that can be easily read on the machine learning stage of our pipeline to train/test models. 

1) OMG_empathy_extract_audio

* Function call: `$ python OMG_empathy_extract_audio.py`
* This function, provided by the organisers of the challenge, generates audio signals from the videos provided.

2) generateListenerVideo

* Function call: `$ python generateListenerVideo.py`
* This function crops the videos provided by the challenge organisers, so they only contain listener's interactions.

3) generateAudioFrames

* Function call: `$ python generateAudioFrames.py`
* This function takes care of segmenting the audios of the interactions, so they can be processed frame by frame.

4) extractSmile

* Function call: `$ python extractSmile.py`
* This function extracts acoustic features from all audio frames, generated by generateAudioFrames.py, in the Audio subfolders using openSMILE.

5) extractFace

* Function call: `$ python extractFace.py`
* This function extracts facial landmarks and facial action units from all videos in VideosListeners folder, with only the listener in each video, using OpenFace.

6) generateFeatureMatrices

* Function call: `$ python generateFeatureMatrices.py dataSplit featureType`

	-> dataSplit: training/validation/test
	
	-> featureType: audio/FAUs/audio+FAUs

* This function generates .csv files with the extracted features and labels, compiled from the individual features extracted previously. The inputs of the function indicate the data partition to generate the files from, and the type of the features to compile.

Finally, the last 2 scripts take care of training and testing the models and generating the predictions in the desired format. The models learnt in both scripts follow different strategies.

7) main_EmpathyPrediction

* Function call: `$ python main_EmpathyPrediction.py -action arg1 -model arg2 -ml arg3 [-dw arg4 -LSTMunits arg5 -bs arg6 -p arg7]`

	-> arg1: train/validate/sanityCheck/train+validate/test
	
 	-> arg2: all/openSMILE/FAUs/openSMILE+FAUs
	
	-> arg3: B-LSTM/2B-LSTM
	
	-> arg4: 1,5
	
	-> arg5: 30,40,50,60,...
	
	-> arg6: 1,5,10,...
	
	-> arg7: 2,3,...

	[ ] indicates optional arguments

* The models trained with this function use the whole dataset available. Therefore, it generates one model at each execution. The argument `-dw` indicates the downsampling factor of the data to be used, `-bs` defines the batch size parameter of the network during training, and `-p` indicates the patience parameter to be used in the early stopping method used for training the network.

8) main_EmpathyPrediction_PersonalizedTrack

* Function call: `$ python main_EmpathyPrediction_PersonalizedTrack.py -action arg1 -model arg2 -ml arg3 [-dw arg4 -LSTMunits arg5 -bs arg6 -p arg7]`

	-> arg1: train/validate/sanityCheck/train+validate/test
	
 	-> arg2: all/openSMILE/FAUs/openSMILE+FAUs
	
	-> arg3: B-LSTM/2B-LSTM
	
	-> arg4: 1,5
	
	-> arg5: 30,40,50,60,...
	
	-> arg6: 1,5,10,...
	
	-> arg7: 2,3,...

	[ ] indicates optional arguments

* The models trained with this function use only intra-subject data. Hence, as many models as subjects in the whole dataset are generated at each execution. The argument `-dw` indicates the downsampling factor of the data to be used, `-bs` defines the batch size parameter of the network during training, and `-p` indicates the patience parameter to be used in the early stopping method used for training the network.

9) calculateCCC

* Function call: `$ python calculateCCC.py annotationsFolder predictionsFolder`
* This function, provided by the challenge organisers, computes the Concordance Correlation Coefficient (CCC) between the annotations and the predictions, whose location must be inputted to the function.
