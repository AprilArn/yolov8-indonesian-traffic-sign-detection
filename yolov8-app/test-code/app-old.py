from flask import Flask, render_template, Response, request, redirect, url_for, session
from ultralytics import YOLO
import cv2
import os
import time
from werkzeug.utils import secure_filename
from flask import jsonify
from flask import g

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session ( CEK ULANG< SEPERTINYA TIDAK PERLU DIPAKAI )

# Folder to save uploaded files
UPLOAD_FOLDER = r'yolov8-app/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model once
model = YOLO(r"D:\Perkuliahan\Semester 8\Skripsi\development\ITSD Project\models\runs_v8n_736\detect\train\weights\best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_camera_stream():
    global latest_detection
    cap = cv2.VideoCapture(1)  # 0 = default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = time.time()

    while True:
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

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Save path in session for access during video_feed
    session['uploaded_video_path'] = filepath
    return redirect(url_for('index'))

def generate_frames(path):
    results = model(path, stream=True)
    prev_time = time.time()

    for result in results:
        frame = result.orig_img
        frame_with_boxes = result.plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.route('/video_feed')
def video_feed():
    video_path = session.get('uploaded_video_path', None)
    if video_path and os.path.exists(video_path):
        return Response(generate_frames(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video uploaded", 400

@app.route('/check_video_uploaded')
def check_video_uploaded():
    video_path = session.get('uploaded_video_path', None)
    if video_path and os.path.exists(video_path):
        return jsonify(status="ok")
    else:
        return jsonify(status="not_found")
    
# Simpan hasil deteksi terakhir (global sementara)
latest_detection = []

def generate_frames(path):
    global latest_detection
    results = model(path, stream=True)
    prev_time = time.time()

    for result in results:
        frame = result.orig_img
        frame_with_boxes = result.plot()

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= 0.65:
                label = model.names[cls_id]
                detections.append(f"{label} ({conf:.2f})")
        
        # Simpan hasil deteksi terbaru
        latest_detection = detections

        # Tambahkan FPS ke frame
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route baru untuk ambil hasil deteksi terbaru
@app.route('/latest_detection')
def get_latest_detection():
    return jsonify(detections=latest_detection)

# Route untuk hapus video yang diupload
@app.route('/clear_uploaded_video', methods=['POST'])
def clear_uploaded_video():
    folder = r'D:\Perkuliahan\Semester 8\Skripsi\development\ITSD Project\yolov8-app\uploads'
    deleted_files = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            os.remove(path)
            deleted_files.append(filename)
        except Exception as e:
            print(f"Gagal hapus {filename}: {e}")

    return jsonify({'status': 'deleted', 'files': deleted_files})

if __name__ == "__main__":
    app.run(debug=True)


# =================================== 


# from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
# from ultralytics import YOLO
# import cv2
# import os
# import time
# from werkzeug.utils import secure_filename
# from lane_detection_module import detect_lane  # 🔁 import fungsi lane detection

# app = Flask(__name__)
# app.secret_key = "your_secret_key"

# UPLOAD_FOLDER = r'yolov8-app/uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Load model YOLOv8
# model = YOLO(r"D:\Perkuliahan\Semester 8\Skripsi\development\ITSD Project\models\runs_v8n_736\detect\train\weights\best.pt")

# latest_detection = []
# previous_lines = None  # 🔁 untuk menyimpan jalur sebelumnya

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/camera_feed')
# def camera_feed():
#     return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_camera_stream():
#     global latest_detection, previous_lines
#     cap = cv2.VideoCapture(1)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     prev_time = time.time()

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         results = model(frame)
#         frame_with_boxes = results[0].plot()

#         # Deteksi rambu
#         detections = []
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             if conf >= 0.65:
#                 label = model.names[cls_id]
#                 detections.append(f"{label} ({conf:.2f})")
#         latest_detection = detections

#         # Deteksi jalur
#         frame_with_boxes, previous_lines = detect_lane(frame_with_boxes, previous_lines)

#         # Tambahkan FPS
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time
#         cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files or request.files['file'].filename == '':
#         return redirect(url_for('index'))

#     file = request.files['file']
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     session['uploaded_video_path'] = filepath
#     return redirect(url_for('index'))

# @app.route('/video_feed')
# def video_feed():
#     video_path = session.get('uploaded_video_path', None)
#     if video_path and os.path.exists(video_path):
#         return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return "No video uploaded", 400

# def generate_frames(path):
#     global latest_detection, previous_lines
#     cap = cv2.VideoCapture(path)
#     prev_time = time.time()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)
#         frame_with_boxes = results[0].plot()

#         # Deteksi rambu
#         detections = []
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             if conf >= 0.65:
#                 label = model.names[cls_id]
#                 detections.append(f"{label} ({conf:.2f})")
#         latest_detection = detections

#         # Deteksi jalur
#         frame_with_boxes, previous_lines = detect_lane(frame_with_boxes, previous_lines)

#         # FPS
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time
#         cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/latest_detection')
# def get_latest_detection():
#     return jsonify(detections=latest_detection)

# @app.route('/check_video_uploaded')
# def check_video_uploaded():
#     video_path = session.get('uploaded_video_path', None)
#     if video_path and os.path.exists(video_path):
#         return jsonify(status="ok")
#     else:
#         return jsonify(status="not_found")

# @app.route('/clear_uploaded_video', methods=['POST'])
# def clear_uploaded_video():
#     folder = UPLOAD_FOLDER
#     deleted_files = []

#     for filename in os.listdir(folder):
#         path = os.path.join(folder, filename)
#         try:
#             os.remove(path)
#             deleted_files.append(filename)
#         except Exception as e:
#             print(f"Gagal hapus {filename}: {e}")

#     return jsonify({'status': 'deleted', 'files': deleted_files})

# if __name__ == "__main__":
#     app.run(debug=True)


# ====================================

