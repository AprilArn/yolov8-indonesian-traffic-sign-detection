from flask import Flask, request, render_template, jsonify
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("D:/Perkuliahan/Semester 8/Skripsi/development/model/runs_640_lr_mom_drop/detect/train/weights/best.pt")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Check file type
        file_ext = os.path.splitext(file.filename)[-1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png']:
            # Process image
            img = cv2.imread(filepath)
            results = model(filepath)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            result_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(result_path, img)

            return jsonify({"result": result_path})
        
        elif file_ext in ['.mp4', '.avi', '.mov']:
            # Process video
            cap = cv2.VideoCapture(filepath)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            result_path = os.path.join(RESULT_FOLDER, file.filename)
            out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())

                        label = f"{model.names[cls]}: {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(frame)

            cap.release()
            out.release()

            return jsonify({"result": result_path})

    return render_template('index_uploads.html')


def process_image(filepath):
    img = cv2.imread(filepath)
    results = model(filepath)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Draw bounding box
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save result
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(filepath))
    cv2.imwrite(result_path, img)
    return result_path

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video setup
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(filepath))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Draw bounding box
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    return result_path

if __name__ == '__main__':
    app.run(debug=True)