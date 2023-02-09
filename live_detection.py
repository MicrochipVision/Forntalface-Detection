import cv2
import os

# Load the Haar cascades for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a folder to save the detected faces
if not os.path.exists('faces'):
    os.makedirs('faces')

# Start the infinite loop
while True:
    # Read the frames from the webcam
    ret, frame = cap.read()
    
    # Convert the frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the detected faces
        roi_gray = gray[y:y+h, x:x+w]
        face_file = "faces/face_{}.png".format(len(os.listdir("faces")))
        cv2.imwrite(face_file, roi_gray)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy the windows
cap.release()
cv2.destroyAllWindows()
