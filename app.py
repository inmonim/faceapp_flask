# -*- coding: utf8 -*-

import os
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2

# =====================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('./model.pt')
model.eval()

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

f = open('name.txt', 'r', encoding='utf-8')
name = f.readline()
name = name.replace('\'', '')
name = name.replace(' ', '')
class_names = name.split(',')
f.close()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_prediction(image_bytes):
    ff = np.fromfile(image_bytes, np.uint8)
    img = cv2.imdecode(ff,cv2.IMREAD_UNCHANGED) # 한글경로 실행법
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    if(len(faces) != 0):
        for (x,y,w,h) in faces:
            cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
            if cropped.size != 0:
                image = torch.Tensor(cropped)
                cv2.imwrite('./crop/crop_img.jpg', cropped)
                image = Image.open('./crop/crop_img.jpg')
                image = transforms_test(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image)
                    _, preds = torch.max(outputs, 1)
                return class_names[preds[0]]
            else:
                return "얼굴 인식 실패"
    else:
        return "얼굴 인식 실패"

#======================================================================

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('faceform.html')


@app.route('/faceapp', methods=['GET', 'POST'])
def mnist():
    if request.method == 'GET':
        return render_template('faceform.html')
    else:
        f = request.files['facefile']
        path = os.path.dirname(__file__)+'/upload/'+f.filename
        f.save(path)
        x = get_prediction(path)
        os.remove(path)

        return render_template('faceresult.html', data=x)

if __name__ == '__main__':
    app.run(debug=True)