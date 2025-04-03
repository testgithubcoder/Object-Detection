import cv2
from ultralytics import YOLO
cap = cv2.VideoCapture(0)
model_path = "yolov8n.pt"  
yolov8_detector = YOLO(model_path)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = yolov8_detector(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {cls}, Conf: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Detected Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
