import cv2
import mediapipe as mp
import pigpio
import time
import numpy as np
import os
import subprocess

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize pigpio
pi = pigpio.pi()
if not pi.connected:
    exit()

# Config
SERVO_PIN = 18
CAMERA_ID = 0
SERVO_MIN_PW = 500
SERVO_MAX_PW = 2500
SERVO_MID_PW = 1500
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SMOOTHING_FACTOR = 0.3
VIDEO_FILE = "Video.mp4"

current_pw = SERVO_MID_PW
pi.set_servo_pulsewidth(SERVO_PIN, current_pw)

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def move_servo_smoothly(target_pw):
    global current_pw
    current_pw = current_pw + SMOOTHING_FACTOR * (target_pw - current_pw)
    current_pw = max(SERVO_MIN_PW, min(SERVO_MAX_PW, current_pw))
    pi.set_servo_pulsewidth(SERVO_PIN, current_pw)

def draw_bounding_box(frame, landmarks, color=(0, 255, 0)):
    h, w, _ = frame.shape
    xs = [lmk.x * w for lmk in landmarks if lmk.visibility > 0.5]
    ys = [lmk.y * h for lmk in landmarks if lmk.visibility > 0.5]
    if xs and ys:
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

def track_person(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    center_x = w // 2
    person_detected = False

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        draw_bounding_box(frame, pose_results.pose_landmarks.landmark)
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * w)
        error = nose_x - center_x
        target_pw = SERVO_MID_PW - error * 5
        move_servo_smoothly(target_pw)
        cv2.line(frame, (nose_x, 30), (nose_x, 60), (0, 255, 0), 2)
        person_detected = True

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return person_detected

def main():
    # Handle existing video file
    if os.path.exists(VIDEO_FILE):
        os.remove(VIDEO_FILE)
        print("Existing Video.mp4 deleted.")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_FILE, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

    print("Recording started. Press 'S' to stop and save video.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            person_detected = track_person(frame)
            status = "Person Detected" if person_detected else "No Person"
            cv2.putText(frame, status, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if person_detected else (0, 0, 255), 2)
            cv2.line(frame, (FRAME_WIDTH//2, 0), (FRAME_WIDTH//2, FRAME_HEIGHT), 
                     (255, 0, 0), 1)

            out.write(frame)
            cv2.imshow("Person Tracking Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                print("Recording stopped.")
                break

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        print("Cleaning up...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        pi.set_servo_pulsewidth(SERVO_PIN, 0)
        pi.stop()

        # Run mocap.py
        print("Running mocap.py...")
        subprocess.run(["python3", "mocap.py"])

if __name__ == "__main__":
    main()
