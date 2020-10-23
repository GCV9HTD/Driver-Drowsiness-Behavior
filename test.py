import os
import ast
import cv2
import glob
import pickle
import math
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from FaceDetector.FaceDetector import faceDetector
from FaceDetector.FeatureExtractor import Extractor

faceobj = faceDetector()
extractor = Extractor()

def calculateDistance(a, b):
    x1, y1 = a
    x2, y2 = b  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist

def extractFeature(src, type_= "dlib"):
    if src.__class__ is str:
        image = cv2.imread(src)
    else:
        image = np.copy(src)

    face_boxes, shape_boxes = faceobj.predict(image, type_=type_)
    shape_predict = None
    if type_ == "dlib":
        X, Y, W, H = 0, 0, 0, 0
        for j, box in enumerate(face_boxes):
            x, y, w, h = box.left(), box.top(), box.width(), box.height()
            if h > H:
                X, Y, W, H = x, y, w, h
                shape_predict = shape_boxes[j]
    else:
        max_ = 0
        for j, box in enumerate(shape_boxes):
            if max_ < calculateDistance(box[1], box[15]):
                shape_predict = box
    if shape_predict is None:
        return None
    feature = np.array(extractor.extract(shape_predict))
    return feature

# def main():
#     model = pickle.load(open("./weights/drowsy.sav", "rb"))
#     cap = cv2.VideoCapture(0)
#     while True:
#         predict = "None"
#         pred = 0
#         ret, frame = cap.read()
#         feature = extractFeature(frame)
#         if feature is not None:
#             feature = feature.reshape(1, -1)
#             pred = model.predict(feature)
#         if pred == 1:
#             predict = "Drowsy"
#         frame = cv2.putText(frame, predict, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,  
#                 1, (0, 255, 0), 1, cv2.LINE_AA)
#         cv2.imshow('live', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows() 

dd = {}
features = []
labels_ = []
tmp = {}

for root, dirs, files in os.walk("dataset/images"):
    for filename in files:
        print (os.path.join(root, filename))