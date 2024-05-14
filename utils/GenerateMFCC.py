import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn

import librosa
import librosa.display
import IPython.display as ipd
import warnings
import librosa
import random
import json
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from IPython.display import Audio
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import os
import math
import json
import random

DATASET_PATH = '../Data/genres_original'
JSON_PATH = '../Data/mfcc/mfcc.json'
SAMPLE_RATE  =  22050
DURATION = 30 #measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE*DURATION

def generateMFCC(dataset_path, json_path, n_mfcc=13, n_fft=4084, hop_length=1024, num_segments=10):
    # dictionary to store data
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': []
    }

    count = 0
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath not in dataset_path:

            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data['mapping'].append(semantic_label)
            print('\nProcessing {}'.format(semantic_label))

            for f in filenames:
                if f.endswith('.wav') and f != 'jazz.00054.wav':  # jazz.00054.wav is an empty file

                    file_path = os.path.join(dirpath, f)

                    # loading the audio file
                    signal, sr = sf.read(file_path)
                    # len(signal) = 661794  # sr is 22050 by default

                    # process segments extracting mfcc and storing data
                    for s in range(num_segments):
                        # Since num_segments is defined as 10. Every 30 sec file is divided into 10 segments of length 3sec
                        # Start sample would keep track of the index of the first element of each 3 second batch
                        # finish sample would keep track of the index of the last element of each 3 second batch
                        # And then with the help of python's slice functionality we will extract that 3 second batch from every 30 sec signal
                        start_sample = num_samples_per_segment * s
                        finish_sample = num_samples_per_segment + start_sample

                        # we need to extract, Usually n_mfcc is set b/w 13 to 40. The other parameters n_fft and hop length are

                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_mfcc=n_mfcc,
                                                    n_fft=n_fft, hop_length=hop_length)

                        mfcc = mfcc.T
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            print(mfcc.shape)
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(i)
                            print('Processing {}, segment:{}'.format(file_path, s))
                            count += 1
                            print(count)

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

generateMFCC(DATASET_PATH, JSON_PATH, num_segments=10)