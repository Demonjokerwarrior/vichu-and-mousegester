import cv2
import os
from datetime import datetime

def create_save_directory(directory_name="recorded_faces"):
    """Create a directory to save the recorded face images."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

def capture_faces(save_directory, face_cascade_path="haarcascade_frontalface_default.xml"):
    """Capture faces from the webcam and save them to the specified directory."""
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit and stop recording.")

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale (needed for Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region
            face_img = frame[y:y + h, x:x + w]

            # Save the face with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(save_directory, f"face_{timestamp}.jpg")
            cv2.imwrite(filename, face_img)

        # Display the frame with detected faces
        cv2.imshow("Face Recording", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the directory to save recorded faces
    save_dir = create_save_directory()

    # Start capturing and saving faces
    capture_faces(save_dir)
