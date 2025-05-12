import cv2
import pickle

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

with open('names.pkl', 'rb') as f:
    names = pickle.load(f)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        if conf < 60:  # (lower = better)
            name = names.get(id, "Unknown")
            label = f"{name} ({round(100 - conf)}%)"
            color = (0, 255, 0)  
        else:
            label = "Unknown Person"
            color = (0, 0, 255)  

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
 
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video.release()
cv2.destroyAllWindows()
