import sys
import cv2
import torch
import torch.nn as nn
from face_detector import face_detector


# specify the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.avgpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.avgpool2(out)
        out = self.cnn3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        out = self.avgpool3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.batchnorm4(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# take in the image ran return a predicted label
def face_recognize(input_image):
    path_model = '../model/model.pkl' # load the saved model
    model = Net()
    model.load_state_dict(torch.load(path_model))
    model.eval()  # change the behavior of the model

    with torch.no_grad():
        inputs = torch.from_numpy(input_image)
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

    return predicted


size = 64
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    faces = face_detector(img)
    for face in faces:
        x, y, w, h = face
        x, y = max(x, 0), max(y, 0)

        img_face = img[y:y + h, x:x + w]
        img_face = cv2.resize(img_face, (size, size))
        img_face = img_face.astype('float32') / 255.0
        img_face = (img_face - 0.5) / 0.5
        img_face = img_face.transpose(2, 0, 1)

        if face_recognize(img_face) == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv2.putText(img, 'Yangwei', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv2.putText(img, 'Others', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        key = cv2.waitKey(1)
        if key == 27:
            sys.exit(0)

    cv2.imshow('Face recognition v2.0', img)

    key = cv2.waitKey(1)
    if key == 27:
        sys.exit(0)
