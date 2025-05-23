import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import pytesseract as pt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

def load_trained_model():
    model = load_model('./object_detection.keras')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def object_detection(model, frame):
    h, w, d = frame.shape
    image_resized = cv2.resize(frame, (224, 224))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make predictions
    coords = model.predict(image_array)[0]
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    return coords

pt.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" #OCR

model = load_trained_model()
registered_plates = pd.read_csv('data/registered_plates.csv')

def is_registered(plate, database):
    return plate.strip().upper() in database['plate'].str.strip().str.upper().values

cap = cv2.VideoCapture(cv2.CAP_DSHOW)


captured_plates = set()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    coords = object_detection(model, frame)
    xmin, xmax, ymin, ymax = coords
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
    roi = frame[ymin:ymax, xmin:xmax]    
    text = pt.image_to_string(roi)    
    registered = is_registered(text, registered_plates)
    box_color = (0, 255, 0) if registered else (255, 0, 0)
    cv2.putText(frame, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
    
    if text.strip() and text not in captured_plates: #new plate
        plate_number = text[:-4].strip() 

        snapshot_filename = f'snapshots/{text.strip().replace(" ", "_")}.png'
        cv2.imwrite(snapshot_filename, frame)
        print(f'Snapshot saved as {snapshot_filename}')
        captured_plates.add(text)

    cv2.imshow('License Plate Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
cap.release()
cv2.destroyAllWindows()

