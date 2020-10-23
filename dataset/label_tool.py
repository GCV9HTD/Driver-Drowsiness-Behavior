import os
import cv2
import pandas as pd

filepaths = []
labels = []

for root, dirs, files in os.walk("./face_cropped"):
    for filename in files:
        filepath = os.path.join(root, filename)
        filepaths.append(filepath)
        image = cv2.imread(filepath)
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord('1'):
            labels.append(1)
        else:
            labels.append(0)
        break

name_dict = {
            'filepath': filepaths,
            'label': labels
          }

df = pd.DataFrame(name_dict)
df.to_csv('drowsy_label.csv')
