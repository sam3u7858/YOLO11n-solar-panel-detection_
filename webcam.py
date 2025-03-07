import cv2
import torch
import time
from ultralytics import YOLO

# Load the trained YOLO model
model_path = "./weights/yolo_nano_solar_panel.pt"
try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open the webcam (0 for default camera, change to a video file path if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get original frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# Initialize FPS calculation variables
prev_frame_time = 0
new_frame_time = 0

# Create a window and allow resizing
cv2.namedWindow("YOLO Live Detection", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.1f}"

        # Perform inference with confidence threshold
        results = model(frame, conf=0.25)  # Adjust confidence threshold as needed

        # Display results on the frame
        for result in results:
            annotated_frame = result.plot()
            
            # Add FPS counter
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show detection counts
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls'):
                num_detections = len(result.boxes.cls)
                cv2.putText(annotated_frame, f"Detections: {num_detections}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Show the processed frame
            cv2.imshow("YOLO Live Detection", annotated_frame)

        # Press 'q' to exit, 's' to save a screenshot
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved as {screenshot_name}")

except KeyboardInterrupt:
    print("Detection stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")