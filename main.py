import ssl
import certifi
import torch
import cv2
import numpy as np

# Configure the SSL context to use certifi's certificate bundle
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Load pre-trained face detection model (Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dummy function for gender prediction (replace with actual model inference)
def predict_gender(face_img):
    # This is a placeholder function. Replace with actual gender prediction code.
    # For example, use your pre-trained model to predict gender from the face image.
    return "Male"  # or "Female" based on your model's prediction

# Function to perform object detection and gender classification using the laptop camera
def detect_objects():
    # Open the laptop camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform object detection inference
        results = model(frame_rgb)
        
        # Render the results on the frame
        frame_with_boxes = results.render()[0]
        
        # Make the frame writable
        frame_with_boxes = np.copy(frame_with_boxes)
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Extract the face from the frame
            face_img = frame_with_boxes[y:y+h, x:x+w]
            
            # Predict gender
            gender = predict_gender(face_img)
            
            # Draw rectangle around the face
            cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Put gender text
            cv2.putText(frame_with_boxes, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Convert the frame back from RGB to BGR
        frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
        
        # Display the resulting frame
        cv2.imshow('Object Detection and Gender Classification', frame_with_boxes)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the object detection and gender classification function
detect_objects()
