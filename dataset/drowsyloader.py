import os 
import cv2
import numpy as np
import pandas as pd

root = "/home/datnvt/Workspace/myself/datnvt/driver drowsiness behavior/dataset"

class DrowsyLoader():
    '''
    Load Drowsiness Behavior dataset
    Return:
        images : ndarray
        labels : ndarray
    '''
    def __init__(self):
        self.csvData = pd.read_csv(os.path.join(root, "drowsy_label.csv"))
        self.filepaths_face = self.csvData["filepath"].values
        self.filepaths_image = [x.replace("face_cropped", "images") for x in self.filepaths_face]
        self.labels = self.csvData["label"].values.reshape(-1, 1)
    
    def load(self):
        images = []
        print ("[INFO] DATASET LOADING...")
        for i, filepath in enumerate(self.filepaths_image):
            img = cv2.imread(os.path.join(root, filepath))
            images.append(img)
            print ('[INFO] DONE READ ', filepath)
        return images, self.labels