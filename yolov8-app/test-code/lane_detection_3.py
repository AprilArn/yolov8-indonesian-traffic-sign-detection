import numpy as np
import cv2

# ================== VIDEO INPUT ==================
video_path = r'D:\Perkuliahan\Semester 8\Skripsi\development\ITSD Project\yolov8-app\uploads\original_test_full_road.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Gagal membuka video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize agar tidak terlalu besar
    frame = cv2.resize(frame, (960, 540))

    # ================== EDGE DETECTION ==================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    # ================== MASKING ==================
    mask = np.zeros_like(edges)
    height, width = edges.shape
    vertices = np.array([[ 
        (int(0.05 * width), height),
        (int(0.95 * width), height),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.4 * width), int(0.6 * height))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    # ================== APPLY MASK ==================
    masked_edges = cv2.bitwise_and(edges, mask)

    # ================== HOUGH TRANSFORM ==================
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=50,
        maxLineGap=150
    )

    # ================== DRAW LINES ==================
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Hindari pembagian nol
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Filter: hanya garis dengan slope curam (tidak horizontal)
            if abs(slope) < 0.5:
                continue

            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    

    # ================== COMBINE ==================
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # ================== TAMPILKAN ==================
    cv2.imshow("Lane Detection - Hough Lines", combo)
    cv2.imshow("Masked Edges", masked_edges)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
