from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import base64
import playsound
import random
import subprocess

app = Flask(__name__, template_folder ='templates')

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    model = YOLO('/Users/jinkwon/Downloads/sleepypy/sleepymodel.pt')

    asleep = 0
    asleep_max = 45

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        box = results[0].boxes
        label = box.cls
        class_labels_strings = [results[0].names[int(labe.cpu().numpy())] for labe in label]
        print(class_labels_strings)

        if "Asleep" in class_labels_strings:
            print("Alert: 'asleep' detected!")
            asleep += 1

            if asleep >= asleep_max:
                asleep = 0
                playsoundPath = playsound.__file__
                otherPython = '/Users/jinkwon/anaconda3/bin/python3.11'
                num = random.randint(1, 2)

                if num == 1:
                    subprocess.check_call(
                        [otherPython, playsoundPath, "/Users/jinkwon/Downloads/sleepypy/mystery1.mp3"])
                else:
                    subprocess.check_call(
                        [otherPython, playsoundPath, "/Users/jinkwon/Downloads/sleepypy/mystery2.mp3"])
        else:
            asleep = 0

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        #frame = base64.b64encode(buffer).decode('utf-8')
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)