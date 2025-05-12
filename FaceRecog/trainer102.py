import cv2
import numpy as np
from PIL import Image
import os
import pickle

path = 'dataface'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# function 
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    names = {} 

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract the name and ID
        name = os.path.split(imagePath)[-1].split(".")[0]
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        if id not in names:
            names[id] = name

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids, names

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids, names = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save to yml
recognizer.write('trainer.yml')  

#  names.pkl
with open('names.pkl', 'wb') as f:
    pickle.dump(names, f)

print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")
