# adapted from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py

import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# if you want to detect yourself, add (img_path, name) tuples to the input_face_pairs list below
input_face_pairs = []
known_face_encodings = []
known_face_names = []

for path, name in input_face_pairs:
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

face_locations = []
face_encodings = []
face_names = []

vid = cv2.VideoCapture(0)
process = 0
while(True):

    ret, frame = video_capture.read()
    if process % 2 == 0:
        rgbframe = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgbframe)
        face_encodings = face_recognition.face_encodings(rgbframe, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    process += 1

vid.release()
cv2.destroyAllWindows()
