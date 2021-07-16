#Import libraries

import nibabel as nib
import numpy as np
import os
from tqdm.notebook import tqdm

#Dataloader Class

class DataLoaderMRILoc:

    def __init__(self, directory, height, width, channels, channels_first):
        self.directory = directory
        self.height = height
        self.width = width
        self.channels = channels
        self.channels_first = channels_first

    def get_data(self):
        files = os.listdir(self.directory) #list files in directory

        #What's the purpose of this line?
        if self.channels_first:
            desired_shape = (self.channels, self.height, self.width)
            desired_shape_opposite = (self.height, self.width, self.channels)
        else:
            desired_shape = (self.height, self.width, self.channels)
            desired_shape_opposite = (self.channels, self.height, self.width)

        X = []
        incorrect_shaped_files = []

        for i, file in enumerate(tqdm(files)):

            image = nib.load(os.path.join(self.directory, file)).get_fdata()

            if image.shape != desired_shape:
                if image.shape == desired_shape_opposite:
                    if self.channels_first:
                        X.append(np.moveaxis(image, -1, 0))
                    else:
                        X.append(np.moveaxis(image, 0, -1))
                else:
                    incorrect_shaped_files.append(file)
                continue
            else:
                X.append(image)

        return np.array(X), incorrect_shaped_files

data_loader = DataLoaderMRILoc('images', 512, 512, 7, False)
X, incorrect_shaped_files = data_loader.get_data()
X.shape
incorrect_shaped_files
