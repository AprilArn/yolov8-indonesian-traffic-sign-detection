from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("D:/Perkuliahan/Semester 8/Skripsi/development/model/runs_640/detect/train/weights/best.pt")

# Path to video
source = "D:/Perkuliahan/Semester 8/Skripsi/development/code/data-test/20250217_135719.mp4"

def generate_frames():
    results = model(source, stream=True)
    prev_time = time.time()

    for result in results:
        frame = result.orig_img
        frame_with_boxes = result.plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)