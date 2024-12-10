import cv2
import face_recognition
import numpy as np
import RPi.GPIO as GPIO
import pickle
import os

BUTTON_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

DATA_FILE = "face_data.pkl"
known_face_encodings = []
known_face_names = []

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as file:
        data = pickle.load(file)
        known_face_encodings = data.get("encodings", [])
        known_face_names = data.get("names", [])

def save_face_data():
    with open(DATA_FILE, "wb") as file:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, file)

def enroll_face():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Position your face in front of the camera and press 's' to save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow("Face Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if face_locations:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                name = input("Enter the person's name: ")
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                save_face_data()
                print(f"Face of {name} enrolled successfully!")
            else:
                print("No face detected. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_faces_for_doorbell():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Button pressed. Checking for faces...")

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        cap.release()
        return

    cap.release()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        if name != "Unknown":
            print(f"Welcome, {name}!")
        else:
            print("Unknown person detected!")

def wait_for_button_press():
    print("Waiting for button press...")
    while True:
        GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING)
        detect_faces_for_doorbell()

if __name__ == "__main__":
    try:
        wait_for_button_press()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

# sudo apt update
# sudo apt install python3-opencv
# pip3 install face_recognition numpy RPi.GPIO
# python3 doorbell.py