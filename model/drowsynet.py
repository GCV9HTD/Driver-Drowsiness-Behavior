import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def drowsynet(num_classes=2):
    model = Sequential()
    model.add(Dense(units= 128, activation="relu",
                    input_shape = (4,)))
    model.add(Dense(units= 128, activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",
                    optimizer=SGD(0.01),
                    metrics= ["accuracy"])
    return model

def drowsyloader():
    csv_data = pd.read_csv("../features/features_s3.csv")
    features = csv_data["feature"].values
    labels = csv_data["label"].values
    X = []
    for feature in features:
        tmp = feature[1: -1].split(' ')
        try:
            feature = np.array(list(map(float, tmp)))
        except:
            feature = []
            for v in tmp:
                if v != '':
                    feature.append(float(v))
            feature = np.array(feature)
        X.append(feature)
    X = np.array(X)
    y = np.array([int(x) for x in labels])
    y = to_categorical(y)
    return X, y

def main():
    X, y = drowsyloader()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = drowsynet()
    # model.fit(X, y, 32, 30)
    # model.save("../weights/drowsynet_final.hdf5")
    model.load_weights("../weights/drowsynet.hdf5")
    score = model.evaluate(X_test, y_test)
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])

# if __name__ == "__main__":
#     main()
