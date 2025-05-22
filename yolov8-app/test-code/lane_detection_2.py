import cv2
import numpy as np

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[ 
        (int(0.05 * width), height),
        (int(0.95 * width), height),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.4 * width), int(0.6 * height))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, polygon

def draw_roi_overlay(frame, polygon):
    overlay = frame.copy()
    cv2.fillPoly(overlay, polygon, (0, 0, 255))  # warna merah (BGR)
    alpha = 0.3  # transparansi
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, polygon, isClosed=True, color=(0, 0, 255), thickness=3)
    return frame

def detect_lines(image):
    lines = cv2.HoughLinesP(image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=80,
                            minLineLength=50,
                            maxLineGap=150)
    return lines

def fit_polynomial_curve_xy(lines):
    if lines is None:
        return None, None

    left_points = []
    right_points = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            slope = 9999
        else:
            slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_points.extend([(y1, x1), (y2, x2)])  # (y, x)
        elif slope > 0.5:
            right_points.extend([(y1, x1), (y2, x2)])

    if len(left_points) < 2 and len(right_points) < 2:
        return None, None

    left_fit = None
    right_fit = None

    if len(left_points) >= 2:
        left_points = np.array(left_points)
        left_fit = np.polyfit(left_points[:,0], left_points[:,1], 2)  # x = f(y)

    if len(right_points) >= 2:
        right_points = np.array(right_points)
        right_fit = np.polyfit(right_points[:,0], right_points[:,1], 2)

    return left_fit, right_fit

def check_curve_slope_limit(fit, slope_threshold=0.8):
    """
    Cek apakah kemiringan (turunan) kurva di sepanjang rentang y melebihi threshold.
    Jika ya, return False (tidak valid)
    """
    if fit is None:
        return False

    # Turunan polinom: 2*a*y + b
    a, b, _ = fit
    # Cek turunan di beberapa titik y (misal 10 titik antara tinggi dan 60% tinggi)
    ys = np.linspace(720, 432, 10)  # contoh asumsi frame tinggi 720px, ROI mulai 60%
    slopes = 2*a*ys + b
    # Karena kita fit x=f(y), slope ini adalah dx/dy
    # Kita bandingkan dengan slope_threshold (misal 0.8)
    if np.any(np.abs(slopes) > slope_threshold):
        return False
    return True

def make_curve_points_xy(image, fit):
    height = image.shape[0]
    plot_y = np.linspace(height, int(height * 0.6), num=50)
    plot_x = np.polyval(fit, plot_y)
    points = np.array([np.array([int(x), int(y)]) for x, y in zip(plot_x, plot_y)])
    return points

def display_curves(image, left_fit, right_fit):
    curve_image = np.zeros_like(image)

    # Cek batas kemiringan dulu
    left_valid = check_curve_slope_limit(left_fit) if left_fit is not None else False
    right_valid = check_curve_slope_limit(right_fit) if right_fit is not None else False

    if left_valid:
        left_points = make_curve_points_xy(image, left_fit)
        cv2.polylines(curve_image, [left_points], isClosed=False, color=(0,255,0), thickness=10)
    if right_valid:
        right_points = make_curve_points_xy(image, right_fit)
        cv2.polylines(curve_image, [right_points], isClosed=False, color=(0,255,0), thickness=10)

    return curve_image

def add_overlay(base_image, line_image):
    return cv2.addWeighted(base_image, 0.8, line_image, 1, 1)

# ================ MAIN =================
cap = cv2.VideoCapture(r'D:\Perkuliahan\Semester 8\Skripsi\development\ITSD Project\yolov8-app\uploads\original_test_full_road.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges = detect_edges(frame)
    roi, polygon = region_of_interest(edges)
    lines = detect_lines(roi)

    frame_with_roi = draw_roi_overlay(frame.copy(), polygon)

    left_fit, right_fit = fit_polynomial_curve_xy(lines)

    if left_fit is not None or right_fit is not None:
        curve_image = display_curves(frame_with_roi, left_fit, right_fit)
        combo_image = add_overlay(frame_with_roi, curve_image)
    else:
        combo_image = frame_with_roi

    cv2.imshow("Lane Detection", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
