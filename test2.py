import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

obama_image = face_recognition.load_image_file("faces/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("faces/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

soham = face_recognition.load_image_file("faces/soham.jpeg")
soham_encode = face_recognition.face_encodings(soham)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    soham_encode
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Borasia sohamsinh"
]

now = datetime.now()
currentdate = now.strftime("%Y-%m-%d")

fs = open(currentdate+'.csv','w+',newline='')
lnwriter = csv.writer(fs)

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(name)
            currenttimee = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, currenttimee])
        cv2.imshow("attandance system", frame)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
fs.close()