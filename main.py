#imports
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


print('Press q to stop webcam recording')

#load model
with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

#set video capture device (using webcam)
capture = cv.VideoCapture(0)

#setup mediapipe feed
with mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as pose:
    while(capture.isOpened):
        ret, frame = capture.read()

        frame = cv.flip(frame,1)

        #Recolor image for media pipe
        image = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        #Detect pose
        result = pose.process(image)
        
        image.flags.writeable = True
        #Recolor image for display
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        

        #Render detections
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            row = np.array([[r.x, r.y, r.z, r.visibility] for r in result.pose_landmarks.landmark]).flatten()
            X = pd.DataFrame([row])
            body_langugage_class = model.predict(X)[0]
            body_langugage_prob = model.predict_proba(X)[0]

            # Draw background rectangle
            cv.rectangle(image, (0,0), (300,80), (245,117,16), -1)

            # Draw 'CLASS' label and value
            cv.putText(image, 'CLASS', (15,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, body_langugage_class.split(' ')[0], (90,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)

            # Draw 'PROB' label and value
            cv.putText(image, 'PROB', (15,60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(round(body_langugage_prob[np.argmax(body_langugage_prob)],2)), (90,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)
            

        except Exception as e:
            print(f"Exception: {e}")

        cv.imshow('Mediapipe Feed', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
capture.release()
cv.destroyAllWindows()