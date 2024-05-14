import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataset import Dataset

class MusicImageDataset(Dataset):
    def __init__(self, data_dir, file_name, flatten=False):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        y_filename (str): Name of file containing labels.
        """
        
        df = pd.read_csv(os.path.join(data_dir, file_name))
        data = df.loc[:, df.columns != 'label'].to_numpy()
        data = torch.from_numpy(data).float()
        #Transform label to number
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(df['label'])
        labels = torch.from_numpy(labels_encoded)

        if flatten:
            self.data = data
        else:
            self.data = data.view((-1, 1,  28, 28))
        self.labels = labels



    def __getitem__(self, index):
        X = self.data[index].float()
        y = self.labels[index]
        return X, y

    def __len__(self):
        return len(self.data)


