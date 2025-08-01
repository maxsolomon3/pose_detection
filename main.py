import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import pandas as pd
import pickle

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

# Load the model
with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    class_signal = pyqtSignal(str)
    prob_signal = pyqtSignal(float)

    def run(self):
        capture = cv.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break

                frame = cv.flip(frame, 1)
                image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image.flags.writeable = False
                result = pose.process(image)
                image.flags.writeable = True
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    try:
                        row = np.array([[r.x, r.y, r.z, r.visibility] for r in result.pose_landmarks.landmark]).flatten()
                        X = pd.DataFrame([row])
                        body_lang_class = model.predict(X)[0]
                        body_lang_prob = model.predict_proba(X)[0]
                        prob_val = round(body_lang_prob[np.argmax(body_lang_prob)], 2)

                        # Emit signals to GUI
                        self.class_signal.emit(body_lang_class)
                        self.prob_signal.emit(prob_val)
                    except Exception as e:
                        print(f"Prediction error: {e}")

                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image.scaled(640, 480))

        capture.release()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deadlift Pose Estimator")

        # Video feed
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        # Classification output labels
        self.class_label = QLabel("Class: ---")
        self.prob_label = QLabel("Probability: ---")

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)

        hbox = QHBoxLayout()
        hbox.addWidget(self.class_label)
        hbox.addWidget(self.prob_label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # Start video thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.class_signal.connect(self.update_class_label)
        self.thread.prob_signal.connect(self.update_prob_label)
        self.thread.start()

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_class_label(self, class_text):
        # Decide arrow direction
        if 'correct' in class_text.lower() or 'up' in class_text.lower():
            arrow = '\u25B2'  # ▲ Up arrow
        elif 'incorrect' in class_text.lower() or 'down' in class_text.lower():
            arrow = '\u25BC'  # ▼ Down arrow
        else:
            arrow = ''  # No arrow if neutral

        self.class_label.setText(f"Class: {class_text} {arrow}")

    def update_prob_label(self, prob_val):
        self.prob_label.setText(f"Probability: {prob_val}")

        # Change color based on range
        if prob_val <= 0.5:
            color = "red"
        elif prob_val <= 0.8:
            color = "orange"
        else:
            color = "green"

        self.prob_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def closeEvent(self, event):
        self.thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())