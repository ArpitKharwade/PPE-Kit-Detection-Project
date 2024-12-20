## Video without alert sound

#******final project********


from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)  # Change camera index if necessary
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)   # Set frame height
# cap.set(3, 640)
# cap.set(4, 480)

# cap = cv2.VideoCapture("../Videos/bikes.mp4") 
# cap = cv2.VideoCapture("../Videos/people.mp4") 

# cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")
# model = YOLO("../Yolo-Weights/yolov8n.pt")
model = YOLO("best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

prev_frame_time = 0
new_frame_time = 0
fps = 0  # Initialize fps variable

# Check if camera is opened
if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
    exit()

while True:
    sucess, img = cap.read()

    if not sucess:
        print("Failed to capture image")
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps = int(fps)  # Convert to integer
    prev_frame_time = new_frame_time

    # Display FPS on the image
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

    print(f'FPS: {fps}')
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()  # Release the camera when done
cv2.destroyAllWindows()  # Close all OpenCV windows
