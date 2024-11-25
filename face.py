import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

def load_known_faces(known_faces_dir):
    """Load known faces and their encodings from a directory."""
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            # Check if a face is detected
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name
            else:
                print(f"No face detected in {filename}. Skipping...")
    return known_face_encodings, known_face_names

def greet_person(name):
    """Greet the recognized person."""
    current_time = datetime.now().strftime("%H:%M:%S")
    greeting = f"Hello, {name}! Welcome. Time: {current_time}"
    print(greeting)

def main():
    # Directory containing images of known faces (update to recorded_faces)
    known_faces_dir = "recorded_faces"  # Referencing the recorded_faces folder
    if not os.path.exists(known_faces_dir):
        print(f"Error: Directory '{known_faces_dir}' does not exist.")
        return

    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    if not known_face_encodings:
        print("No known faces found in the directory. Exiting...")
        return

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    print("Press 'q' to quit the program.")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                greet_person(name)

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
