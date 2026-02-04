import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture("anh/bang_chuyen.mp4")
if not cap.isOpened():
    print("Không mở được video")
    exit()

# ====== TÌM VẠCH ĐỎ ======
ret, first_frame = cap.read()
hsv = cv.cvtColor(first_frame, cv.COLOR_BGR2HSV)

lower_red1 = np.array([0,120,70])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([170,120,70])
upper_red2 = np.array([180,255,255])

mask = cv.inRange(hsv, lower_red1, upper_red1) + cv.inRange(hsv, lower_red2, upper_red2)
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

line_X = None
for c in contours:
    if cv.contourArea(c) > 500:
        x,y,w,h = cv.boundingRect(c)
        line_X = x + w//2
        break

if line_X is None:
    print("Không tìm thấy vạch đỏ")
    exit()

print("Vạch đỏ tại X =", line_X)
cap.set(cv.CAP_PROP_POS_FRAMES, 1)

# ====== TRACKING ======
next_id = 0
objects = {}   # id -> (x, y)
counted_ids = set()
count = 0

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=30,
                              param1=50, param2=30, minRadius=10, maxRadius=50)

    detected_centers = []

    if circles is not None:
        circles = np.uint16(np.round(circles))
        for c in circles[0, :]:
            detected_centers.append((c[0], c[1], c[2]))  # x,y,r

    new_objects = {}

    # ====== GÁN ID ======
    for (x, y, r) in detected_centers:
        matched_id = None
        for obj_id, (px, py) in objects.items():
            if distance((x, y), (px, py)) < 40:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        new_objects[matched_id] = (x, y)

        # ====== ĐẾM KHI QUA VẠCH ======
        if matched_id not in counted_ids:
            if objects.get(matched_id) is not None:
                old_x = objects[matched_id][0]
                if old_x < line_X and x >= line_X:
                    count += 1
                    counted_ids.add(matched_id)

        cv.circle(frame, (x, y), r, (0,0,255), 2)
        cv.putText(frame, f"ID {matched_id}", (x-10, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    objects = new_objects

    cv.line(frame, (line_X, 0), (line_X, frame.shape[0]), (0,255,255), 2)
    cv.putText(frame, f"Count: {count}", (20,50),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

    cv.imshow("Tracking Count", frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

print("TỔNG SỐ VẬT QUA VẠCH:", count)

cap.release()
cv.destroyAllWindows()
