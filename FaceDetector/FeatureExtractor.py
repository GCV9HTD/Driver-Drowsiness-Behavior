import math
from scipy.spatial import distance
from math import atan2,degrees

class Extractor():
    def __init__(self):
        pass

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = distance.euclidean(mouth[14], mouth[18])
        C = distance.euclidean(mouth[12], mouth[16])
        mar = (A ) / (C)
        return mar

    def circularity(self, eye):
        A = distance.euclidean(eye[1], eye[4])
        radius  = A/2.0
        Area = math.pi * (radius ** 2)
        p = 0
        p += distance.euclidean(eye[0], eye[1])
        p += distance.euclidean(eye[1], eye[2])
        p += distance.euclidean(eye[2], eye[3])
        p += distance.euclidean(eye[3], eye[4])
        p += distance.euclidean(eye[4], eye[5])
        p += distance.euclidean(eye[5], eye[0])
        return 4 * math.pi * Area /(p**2)

    def mouth_over_eye(self, eye):
        ear = self.eye_aspect_ratio(eye)
        mar = self.mouth_aspect_ratio(eye)
        mouth_eye = mar/ear
        return mouth_eye


    def average(self, y_pred):
        for i in range(len(y_pred)):
            if i % 240 == 0 or (i+1) % 240 == 0:
                pass
            else: 
                average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
                if average >= 0.5:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        return y_pred

    def extract(self, landmarks):
        eye = landmarks[36: 68]
        EAR = self.eye_aspect_ratio(eye)
        MAR = self.mouth_aspect_ratio(eye)
        CIR = self.circularity(eye)
        MOE = self.mouth_over_eye(eye)
        return [EAR, MAR, CIR, MOE]
    

    
