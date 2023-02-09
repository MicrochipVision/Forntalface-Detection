import cv2
import os

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the input image
img = cv2.imread("input.jpg")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Crop and save the faces
if len(faces) > 0:
    os.makedirs("faces", exist_ok=True)
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150))
        cv2.imwrite(f"faces/face_{i}.jpg", face)

# Draw rectangles around the faces and display the output
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Resize the image to display it in a smaller window
img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
cv2.imshow("Faces", img)
cv2.waitKey()
cv2.destroyAllWindows()
