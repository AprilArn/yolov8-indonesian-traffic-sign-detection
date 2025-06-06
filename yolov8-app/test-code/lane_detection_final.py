import cv2
import numpy as np
import time

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[ 
        (int(0.025 * width), height),
        (int(0.975 * width), height),
        (int(0.6 * width), int(0.55 * height)),
        (int(0.4 * width), int(0.55 * height))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, polygon

def draw_roi_overlay(frame, polygon):
    overlay = frame.copy()
    cv2.fillPoly(overlay, polygon, (0, 0, 0))  # warna hitam (BGR)
    alpha = 0.6  # transparansi
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, polygon, isClosed=True, color=(20, 20, 20), thickness=2)
    return frame

def detect_lines(image):
    lines = cv2.HoughLinesP(image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=78,
                            minLineLength=40,
                            maxLineGap=80)
    return lines

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0] * 0.85) # move y1 to the bottom of the image
    y2 = int(image.shape[0] * 0.7) # move y2 to the top of the image
    if slope == 0:
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    width = image.shape[1]
    center_x = width // 2

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters

        if slope < -0.5 and x1 < center_x and x2 < center_x:
            left_fit.append((slope, intercept))
        elif slope > 0.5 and x1 > center_x and x2 > center_x:
            right_fit.append((slope, intercept))

    if len(left_fit) == 0 or len(right_fit) == 0:
        return None

    left_avg = np.average(left_fit, axis=0)
    right_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_avg)
    right_line = make_coordinates(image, right_avg)

    return np.array([left_line, right_line])

def lerp_lines(old_lines, new_lines, alpha=0.1):
    if old_lines is None:
        return new_lines
    smoothed = []
    for old, new in zip(old_lines, new_lines):
        smoothed_line = old * (1 - alpha) + new * alpha
        smoothed.append(smoothed_line.astype(int))
    return np.array(smoothed)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return line_image

def add_overlay(base_image, line_image):
    return cv2.addWeighted(base_image, 1, line_image, 1, 1)

# ================ MAIN =================
cap = cv2.VideoCapture(r'D:\Perkuliahan\Semester 8\Skripsi\development\ITSD Project\yolov8-app\uploads\original_test_full_road.mp4')

previous_lines = None
last_detection_time = 0
max_no_detection_duration = 0.5  # detik

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    edges = detect_edges(frame)
    roi, polygon = region_of_interest(edges)
    lines = detect_lines(roi)
    averaged_lines = average_slope_intercept(frame, lines)

    if averaged_lines is not None:
        if previous_lines is None:
            previous_lines = averaged_lines
        else:
            previous_lines = lerp_lines(previous_lines, averaged_lines, alpha=0.1)
        last_detection_time = current_time
    elif current_time - last_detection_time > max_no_detection_duration:
        previous_lines = None

    frame_with_roi = draw_roi_overlay(frame.copy(), polygon)

    if previous_lines is not None:
        line_image = display_lines(frame_with_roi, previous_lines)
        combo_image = add_overlay(frame_with_roi, line_image)
    else:
        combo_image = frame_with_roi

    cv2.imshow("Lane Detection", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
