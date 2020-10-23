import os
import ast
import cv2
import glob
import pickle
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from FaceDetector.FaceDetector import faceDetector
from FaceDetector.FeatureExtractor import Extractor
from dataset.drowsyloader import DrowsyLoader

faceobj = faceDetector()
extractor = Extractor()
loader = DrowsyLoader()

def average(y_pred):
    for i in range(len(y_pred) - 1):
        if i % 240 == 0 or (i+1) % 240 == 0:
            pass
        else: 
            average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
            if average >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
    return y_pred

def extractFeature():
    features = []
    images, labels = loader.load()
    for i, image in enumerate(images):
        face_boxes, shape_boxes = faceobj.predict(image)
        X, Y, W, H = 0, 0, 0, 0
        shape_predict = None
        for j, box in enumerate(face_boxes):
            x, y, w, h = box.left(), box.top(), box.width(), box.height()
            if h > H:
                X, Y, W, H = x, y, w, h
                shape_predict = shape_boxes[j]
        features.append(extractor.extract(shape_predict))
        print (i+1)
    dict_name = {
        "feature" : features,
        "label" : list(labels)
    }
    pf = pd.DataFrame(dict_name)
    pf.to_csv("../features/features.csv")

def KNN():
    data = pd.read_csv("../features/features.csv")
    X = data["feature"].values
    y = data["label"].values
    convert = lambda arr : np.fromstring(arr[1:-1], dtype=np.float, sep=',')
    X = np.array([convert(x) for x in X])
    y = np.array([int(convert(x)) for x in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    acc3_list = []
    f1_score3_list = []
    roc_3_list = []
    for i in range(1,30):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, y_train) 
        pred_KN = neigh.predict(X_test)
        pred_KN = average(pred_KN)
        y_score_3 = neigh.predict_proba(X_test)[:,1]
        acc3_list.append(accuracy_score(y_test, pred_KN))
        f1_score3_list.append(metrics.f1_score(y_test, pred_KN))
        roc_3_list.append(metrics.roc_auc_score(y_test, y_score_3))
        
    # 16
    neigh = KNeighborsClassifier(n_neighbors=acc3_list.index(max(acc3_list))+1)
    neigh.fit(X_train, y_train) 
    filename = '../weights/drowsy.sav'
    pickle.dump(neigh, open(filename, 'wb'))
    pred_KN = neigh.predict(X_test)
    pred_KN = average(pred_KN)
    y_score_3 = neigh.predict_proba(X_test)[:,1]
    acc3 = accuracy_score(y_test, pred_KN)
    f1_score_3 = metrics.f1_score(y_test, pred_KN)
    roc_3 = metrics.roc_auc_score(y_test, y_score_3)

    print([acc3,f1_score_3,roc_3])
    print(confusion_matrix(y_test, pred_KN))

def main():
    KNN()

    


if __name__ == "__main__":
    main()