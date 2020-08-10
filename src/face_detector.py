# Standard library imports
from pathlib import Path
import glob
import time
import numpy as np
import cv2

#top_level_dir path
root_path = Path(__file__).parent.parent

class FaceDetector:
    """
    This class is used for detecting face.
    """

    def __init__(self):

        """
        Constructor of class
        """

        config_path = root_path.joinpath("pretrained/",
                                "resnet10_ssd.prototxt")
        face_model_path = root_path.joinpath("pretrained/",
                "res10_300x300_ssd_iter_140000.caffemodel")

        self.detector = cv2.dnn.readNetFromCaffe(str(config_path),
                                            str(face_model_path))

        #detector prediction threshold
        self.confidence = 0.7


    def get(self,img):
        """
        Given a image, detect faces and compute their bb

        """
        bb =  self._detect_face_ResNet10_SSD(img)

        return bb

    def _detect_face_ResNet10_SSD(self,img):
        """
        Given a img, detect faces in it using resnet10_ssd detector

        """

        detector = self.detector
        (h, w) = img.shape[:2]
        # construct a blob from the image
        img_blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(img_blob)
        detections = detector.forward()

        (start_x, start_y, end_x, end_y) = (0,0,0,0)
        faces_bb = []
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            for i in range(0, detections.shape[2]):

                score = detections[0, 0, i, 2]

                # ensure that the detection greater than our threshold is
                # selected
                if score > self.confidence:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    (start_x, start_y, end_x, end_y) = box

                    # extract the face ROI and grab the ROI dimensions
                    face = img[start_y:end_y, start_x:end_x]

                    (fh, fw) = face.shape[:2]
                    # ensure the face width and height are sufficiently large
                    if fw < 20 or fh < 20:
                        pass
                    else:
                        faces_bb.append(box)

        if(len(faces_bb)>0):
            faces_bb = np.array(faces_bb)

        return faces_bb
