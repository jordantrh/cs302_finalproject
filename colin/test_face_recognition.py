# adapted from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py

import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# if you want to detect yourself, add (img_path, name) tuples to the input_face_pairs list below
input_face_pairs = [('colin.jpg','colin')]
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
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
vid = cv2.VideoCapture(0)
process = 0
while(True):

    ret, frame = video_capture.read()
    if process % 4 == 0:
        rgbframe = frame[:, :, ::-1]
        #face_locations = face_recognition.face_locations(rgbframe)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = faceCascade.detectMultiscale(gray,1.1,4)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        face_locations = [(x[1], x[0]+x[2], x[1]+x[3], x[0]) for x in faces]
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
    '''
    for(x,y,w,h) in face_locations:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    for (x, y, w, h), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)'''

    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    process += 1

vid.release()
cv2.destroyAllWindows()
