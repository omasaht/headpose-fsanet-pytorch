"""
Dataset Class for FSANet Training
Implemented by Omar Hassan
August, 2020
"""

from torch.utils.data import Dataset
import torch
import numpy as np
import glob

class HeadposeDataset(Dataset):

    def __init__(self,data_path,
                transform=None):

        self.transform = transform

        #since the data is not much, we can load it
        #entirely in RAM
        files_path = glob.glob(f'{data_path}/*.npz')
        image = []
        pose = []
        for path in files_path:
            data = np.load(path)
            image.append(data["image"])
            pose.append(data["pose"])

        image = np.concatenate(image,0)
        pose = np.concatenate(pose,0)

        #exclude examples with pose outside [-99,99]
        x_data = []
        y_data = []
        for i in range(pose.shape[0]):
            if np.max(pose[i,:])<=99.0 and np.min(pose[i,:])>=-99.0:
                x_data.append(image[i])
                y_data.append(pose[i])

        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)


        print('x (images) shape: ',self.x_data.shape)
        print('y (poses) shape: ',self.y_data.shape)

    def set_transform(self,transform):
        self.transform = transform

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        if(self.transform):
            x = self.transform(x)

        return x,y

#used to apply different transforms to train,validation dataset
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
