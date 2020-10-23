import cv2
import dlib
import imutils
import face_alignment
from imutils import face_utils

class faceDetector():
    '''
    Description:
        Detect face and 68 landmarks point in image
    Return:
        bbox faces and set 68 landmarks all face in image
    '''
    def __init__(self):
        '''
        Description:
            Initial class, detector and predictor
        '''
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./FaceDetector/shape_predictor/shape_predictor_68_face_landmarks.dat")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    def predict(self, path, type_="dlib"):
        '''
        Description:
            Predict face boxes and shape boxes
        Parameters:
            path: string or ndarray
        '''
        if path.__class__ is str:
            image = cv2.imread(path)
        else:
            image = path.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        face_boxes = []
        shape_boxes = []
        if type_ == "dlib":
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                face_boxes.append(rect)
                shape_boxes.append(shape)
        else:
            shape_boxes = self.fa.get_landmarks(image)
        return face_boxes, shape_boxes

