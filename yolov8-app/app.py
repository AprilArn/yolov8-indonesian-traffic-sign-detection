from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from ultralytics import YOLO
import cv2
import os
import time
from werkzeug.utils import secure_filename
from lane_detection_module import detect_lane

app = Flask(__name__)
app.secret_key = "your_secret_key"  # maybe i don't need this

UPLOAD_FOLDER = r'yolov8-app/media/uploads'
RESULT_FOLDER = r'yolov8-app/media'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(r"D:\Perkuliahan\Semester 8\Skripsi\version-control\yolov8-indonesian-traffic-sign-detection\models\runs_v8n_736\detect\train\weights\best.pt")

current_fps = 0

latest_detection = []
previous_lines = None
last_detection_time = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_fps')
def get_fps():
    return jsonify(fps=round(current_fps, 2))

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_camera_stream():
    global latest_detection, previous_lines, last_detection_time
    cap = cv2.VideoCapture(1)
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        frame_with_boxes = results[0].plot()

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= 0.65:
                label = model.names[cls_id]
                detections.append(f"{label} ({conf:.2f})")
        latest_detection = detections

        frame_with_boxes, previous_lines, last_detection_time = detect_lane(
            frame_with_boxes, previous_lines, last_detection_time
        )

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        global current_fps
        current_fps = fps


        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    session['uploaded_video_path'] = filepath
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    video_path = session.get('uploaded_video_path')
    if video_path and os.path.exists(video_path):
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video uploaded", 400

def generate_frames(path):
    global latest_detection, previous_lines, last_detection_time
    cap = cv2.VideoCapture(path)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_with_boxes = results[0].plot()

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= 0.7:
                label = model.names[cls_id]
                detections.append(f"{label} ({conf:.2f})")
        latest_detection = detections

        frame_with_boxes, previous_lines, last_detection_time = detect_lane(
            frame_with_boxes, previous_lines, last_detection_time
        )

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        global current_fps
        current_fps = fps

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/latest_detection')
def get_latest_detection():
    return jsonify(detections=latest_detection)

@app.route('/check_video_uploaded')
def check_video_uploaded():
    video_path = session.get('uploaded_video_path')
    if video_path and os.path.exists(video_path):
        return jsonify(status="ok")
    return jsonify(status="not_found")

@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    video_path = session.get('uploaded_video_path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({"status": "error", "message": "No uploaded video found."}), 400

    output_path = os.path.join(RESULT_FOLDER, 'result_' + os.path.basename(video_path))

    results = model.predict(
        source=video_path,
        save=True,
        save_txt=False,
        save_conf=True,
        project=RESULT_FOLDER,
        name='',
        exist_ok=True
    )

    return jsonify({"status": "ok", "message": "Prediction saved.", "output": output_path})

@app.route('/clear_uploaded_video', methods=['POST'])
def clear_uploaded_video():
    deleted_files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            os.remove(path)
            deleted_files.append(filename)
        except Exception as e:
            print(f"Gagal hapus {filename}: {e}")
    return jsonify({'status': 'deleted', 'files': deleted_files})

if __name__ == "__main__":
    app.run(debug=True)
