#!/usr/bin/python2
# Python 2 script

import os

# OpenSmile - https://www.audeering.com/technology/opensmile/
OpenSmilePath = '/tools/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
OpenSmileConf = '-C /tools/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -timestampcsv 0 -headercsv 0 -appendcsv 0'  # openSMILE configuation (eGeMAPS)


# Path of audio and filenames
audio_folders = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/Audio/',
                    '/nas/staff/data_work/Adria/OMG_Challenge/Validation/Audio/',
                    '/nas/staff/data_work/Adria/OMG_Challenge/Test/Audio/']

output_folders = ['/nas/staff/data_work/Adria/OMG_Challenge/Training/AudioFeatures/',
                    '/nas/staff/data_work/Adria/OMG_Challenge/Validation/AudioFeatures/',
                    '/nas/staff/data_work/Adria/OMG_Challenge/Test/AudioFeatures/']

# Extract acoustic features with openSMILE
for folder_in, folder_out in zip(audio_folders, output_folders):
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    
    subfolders = [folder for folder in os.listdir(folder_in) if os.path.isdir(folder_in + folder)] 
    
    for sf in subfolders:

        if (len(sf.split('_')) == 5) and (sf.split('_')[4] == 'downsampledDataBy5'):

            sf = sf + '/'
            if not os.path.exists(folder_out + sf):
                os.mkdir(folder_out + sf)
            
            for fn in os.listdir(folder_in + sf):
                if fn.endswith('.wav'):
                    os.system(OpenSmilePath + ' ' + OpenSmileConf + ' -I ' + folder_in + sf + fn + ' -instname ' + sf + fn + ' -csvoutput  ' + folder_out + sf + fn[:-3] + 'csv')
    
