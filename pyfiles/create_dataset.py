import os
import sys
import cv2
from numpy import random
from face_detector import face_detector


# randomly change the brightness and contrast of the image to augment the data
def img_change(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, k] = tmp
    return img


# read in the webcam video stream
# detect my face(with or without mask) and save it into target folder
def save_myfaces_masked():
    result_path = '../data/faces_my_masked'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    size = 64
    num_images = 1000
    cnt = 1

    cap = cv2.VideoCapture(0)
    while True:
        if cnt <= num_images:
            print('Being processed picture %s' % cnt)
            _, img = cap.read()

            faces = face_detector(img)
            for face in faces:
                x, y, w, h = face
                x, y = max(x, 0), max(y, 0)
                long_side = max(w, h)

                img_face = img[y:y + long_side, x:x + long_side]
                img_face = cv2.resize(img_face, (size, size))
                img_face = img_change(img_face, random.uniform(0.5, 1.5), random.randint(-50, 50))

                cv2.imshow('image', img_face)
                cv2.imwrite(result_path + '/' + str(cnt + 1000) + '.jpg', img_face)
                cnt += 1

            key = cv2.waitKey(50)
            if key == 27:
                break

        else:
            print('Finished!')
            break


# the original images are already faces
# only resize them to fit the model
def save_other_masked():
    source_path = '../data/faces_raw'
    result_path = '../data/faces_other_masked'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    size = 64
    cnt = 1

    for (path, dirnames, filenames) in os.walk(source_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                print('Being processed picture %s' % cnt)
                img_path = path + '/' + filename

                img_face = cv2.imread(img_path)
                img_face = cv2.resize(img_face, (size, size))
                cv2.imshow('image', img_face)
                cv2.imwrite(result_path + '/' + str(cnt) + '.jpg', img_face)  # save face
                cnt += 1

                key = cv2.waitKey(10)
                if key == 27:
                    sys.exit(0)

# call the above function to create custom datasets
# save_myfaces_masked()
# save_other_masked()
