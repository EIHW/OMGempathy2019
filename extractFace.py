#!/usr/bin/python2
# Python 2 script

import os

# OpenFace https://github.com/TadasBaltrusaitis/OpenFace/
OpenFacePath = '/tools/OpenFace/build/bin/FeatureExtraction'
OpenFaceConf = '-2Dfp -aus -tracked' # Facial Landmarks & FAUs


# Path of videos (ONLY one subject present throughout)
video_folders = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/VideosListeners/',
                 	'/nas/staff/data_work/Adria/OMG_Challenge/Validation/VideosListeners/',
					'/nas/staff/data_work/Adria/OMG_Challenge/Test/VideosListeners/']

output_folders = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/VideoFeatures/',
	                '/nas/staff/data_work/Adria/OMG_Challenge/Validation/VideoFeatures/',
					'/nas/staff/data_work/Adria/OMG_Challenge/Test/VideoFeatures/']

# Extract and convert visual features with openFACE
for folder_in, folder_out in zip(video_folders, output_folders):
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    
    for fn in os.listdir(folder_in):
        if fn.endswith('.mp4'):
            os.system(OpenFacePath + ' ' + OpenFaceConf + ' -f ' + folder_in + fn + ' -out_dir ' + folder_out)

