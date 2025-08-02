import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import pandas as pd
import pickle

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QFormLayout
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
    rep_signal = pyqtSignal(int)

    def run(self):
        self.reps = 0
        self.rep_state = None  # 'up' or 'down'

        capture = cv.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
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

                    # Rep counting logic (prob value used for smoothing between reps)
                    if body_lang_class == "up" and prob_val > 0.7:
                        if self.rep_state == "down":
                            self.reps += 1
                            self.rep_signal.emit(self.reps)
                        self.rep_state = "up"
                    else:
                        self.rep_state = "down"

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

        # Video feed display
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        # Classification output labels
        self.class_label = QLabel("Class: ---")
        self.prob_label = QLabel("Probability: ---")

        # Repetition counter
        self.rep_label = QLabel("Reps: 0")
        self.rep_count = 0

        # Weight input field
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("e.g. 135")

        # Layouts
        vbox = QVBoxLayout()

        # Top-right layout for reps and weight input
        top_right_layout = QHBoxLayout()
        top_right_layout.addWidget(self.rep_label)

        weight_form = QFormLayout()
        weight_form.addRow("Weight (lbs):", self.weight_input)
        top_right_layout.addLayout(weight_form)
        top_right_layout.addStretch()  # Push to top-right

        vbox.addLayout(top_right_layout)
        vbox.addWidget(self.image_label)

        # Bottom layout for class and prob
        bottom_hbox = QHBoxLayout()
        bottom_hbox.addWidget(self.class_label)
        bottom_hbox.addWidget(self.prob_label)
        vbox.addLayout(bottom_hbox)

        self.setLayout(vbox)

        # Start video thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.class_signal.connect(self.update_class_label)
        self.thread.prob_signal.connect(self.update_prob_label)
        self.thread.rep_signal.connect(self.update_rep_label)
        self.thread.start()

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_class_label(self, class_text):
        if 'correct' in class_text.lower() or 'up' in class_text.lower():
            arrow = '\u25B2'  # ▲ Up
        elif 'incorrect' in class_text.lower() or 'down' in class_text.lower():
            arrow = '\u25BC'  # ▼ Down
        else:
            arrow = ''
        self.class_label.setText(f"Class: {class_text} {arrow}")

    def update_prob_label(self, prob_val):
        self.prob_label.setText(f"Probability: {prob_val}")
        if prob_val <= 0.5:
            color = "red"
        elif prob_val <= 0.8:
            color = "orange"
        else:
            color = "green"
        self.prob_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def update_rep_label(self, count):
        self.rep_count = count
        self.rep_label.setText(f"Reps: {count}")

    def closeEvent(self, event):
        self.thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())