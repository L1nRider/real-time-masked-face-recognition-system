import cv2
import numpy as np

# These two directories need to be in absolute format
# you can find these two files under ../common
path_model = "../common/deploy.prototxt.txt"
path_weight = "../common/res10_300x300_ssd_iter_140000.caffemodel"


# detect face in the input image
# return the upper left (x,y), width and height
# can detect multiple faces
def face_detector(img):
    net = cv2.dnn.readNetFromCaffe(path_model, path_weight)  # call OpenCV pretrained DNN model
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    threshold = 0.5
    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x_start, y_start, x_end, y_end = box.astype("int")
        faces.append(np.array([x_start, y_start, x_end - x_start, y_end - y_start]))

    return faces
