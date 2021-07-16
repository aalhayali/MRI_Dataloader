# Import libraries

import nibabel as nib
import numpy as np
import os
from tqdm.notebook import tqdm

# Dataloader Class

class DataLoaderMRILoc:

    def __init__(self, directory, height, width, channels, channels_first):
        
        # Directory of all the images
        self.directory = directory
        
        # Image Attributes
        self.height = height
        self.width = width
        self.channels = channels
        
        # Format of the image (Bool)
        self.channels_first = channels_first
        
        # Desired shape of the volume along with the opposite
        # This would check for both channel first and channel last images and convert them to the specified format
        self.desired_shape = (self.channels, self.height, self.width) if self.channels_first else (self.height, self.width, self.channels)
        self.desired_shape_opposite = (self.height, self.width, self.channels) if self.channels_first else (self.channels, self.height, self.width)

    def get_data(self):
        
        # List files in directory
        files = os.listdir(self.directory)
        
        # Collect all numpy arrays
        X = []
        
        # Names of files whose shape is other than the desired shape (except channel positions)
        incorrect_shaped_files = []

        for i, file in enumerate(tqdm(files)):

            image = nib.load(os.path.join(self.directory, file)).get_fdata()

            if image.shape != self.desired_shape:
                if image.shape == self.desired_shape_opposite:
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


    
# Example
directory = 'images'
height = 512
width = 512
channels = 7
channels_first = False

data_loader = DataLoaderMRILoc(directory, height, width, channels, channels_first)
X, incorrect_shaped_files = data_loader.get_data()

# Checking
print(X.shape)
print(incorrect_shaped_files)
