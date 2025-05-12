import cv2
import os
import pickle

dataset_folder = 'dataface'
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

id_registry_path = 'id_registry.pkl'
id_registry = {}

if os.path.exists(id_registry_path):
    try:
        with open(id_registry_path, 'rb') as f:
            id_registry = pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load id_registry: {e}")
        print("[INFO] Reinitializing id registry...")
        id_registry = {}

name = input("Enter the name of the person: ").strip().lower()

if name in id_registry:
    person_id = id_registry[name]
else:
    person_id = max(id_registry.values(), default=0) + 1
    id_registry[name] = person_id
    with open(id_registry_path, 'wb') as f:
        pickle.dump(id_registry, f)

print(f"[INFO] Assigned ID {person_id} to '{name}'")

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        filename = f'{dataset_folder}/{name}.{person_id}.{count}.jpg'
        cv2.imwrite(filename, gray[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Capturing Faces", frame)

    if count >= 50:
        print("[INFO] Dataset collection complete.")
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("[INFO] Stopped by user.")
        break

video.release()
cv2.destroyAllWindows()
