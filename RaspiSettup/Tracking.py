import cv2
import mediapipe as mp
import pigpio
import time
import numpy as np
import os
import subprocess
import threading

#Make the flutter app run this script , this script starts tracking humans detected

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize pigpio
pi = pigpio.pi()
if not pi.connected:
    print("ERROR: pigpio connection failed")
    exit()

# Config
SERVO_PAN_PIN = 18
SERVO_TILT_PIN = 19  # Added tilt servo
CAMERA_ID = 0
SERVO_MIN_PW = 500
SERVO_MAX_PW = 2500
SERVO_MID_PW = 1500
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SMOOTHING_FACTOR = 0.3
VIDEO_FILE = "Video.mp4"

# Initialize servo positions
current_pan_pw = SERVO_MID_PW
current_tilt_pw = SERVO_MID_PW
pi.set_servo_pulsewidth(SERVO_PAN_PIN, current_pan_pw)
pi.set_servo_pulsewidth(SERVO_TILT_PIN, current_tilt_pw)

# Status flags
camera_connected = False
person_visible = False
recording = False

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def move_servo_smoothly(servo_pin, current_pw, target_pw):
    new_pw = current_pw + SMOOTHING_FACTOR * (target_pw - current_pw)
    new_pw = max(SERVO_MIN_PW, min(SERVO_MAX_PW, new_pw))
    pi.set_servo_pulsewidth(servo_pin, new_pw)
    return new_pw

def draw_bounding_box(frame, landmarks, color=(0, 255, 0)):
    h, w, _ = frame.shape
    xs = [lmk.x * w for lmk in landmarks if lmk.visibility > 0.5]
    ys = [lmk.y * h for lmk in landmarks if lmk.visibility > 0.5]
    if xs and ys:
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        return (x_min, y_min, x_max, y_max)
    return None

def track_person(frame):
    global current_pan_pw, current_tilt_pw, person_visible
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    center_x = w // 2
    center_y = h // 2
    person_detected = False

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        bbox = draw_bounding_box(frame, pose_results.pose_landmarks.landmark)
        
        # Use nose for tracking
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        
        # Calculate error from center
        x_error = nose_x - center_x
        y_error = nose_y - center_y
        
        # Calculate target servo positions
        target_pan_pw = SERVO_MID_PW - x_error * 5  # Invert for correct direction
        target_tilt_pw = SERVO_MID_PW + y_error * 5  # May need adjustment based on servo mounting
        
        # Move servos smoothly
        current_pan_pw = move_servo_smoothly(SERVO_PAN_PIN, current_pan_pw, target_pan_pw)
        current_tilt_pw = move_servo_smoothly(SERVO_TILT_PIN, current_tilt_pw, target_tilt_pw)
        
        # Draw tracking indicators
        cv2.line(frame, (nose_x, 30), (nose_x, 60), (0, 255, 0), 2)
        cv2.line(frame, (30, nose_y), (60, nose_y), (0, 255, 0), 2)
        
        # Print tracking coordinates
        print(f"Tracking human at coordinates: X={nose_x}, Y={nose_y} | Servo PAN: {current_pan_pw}, TILT: {current_tilt_pw}")
        
        person_detected = True
        person_visible = True

    elif hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand center for tracking
            xs = [lmk.x * w for lmk in hand_landmarks.landmark]
            ys = [lmk.y * h for lmk in hand_landmarks.landmark]
            hand_x = int(sum(xs) / len(xs))
            hand_y = int(sum(ys) / len(ys))
            
            # Calculate error from center
            x_error = hand_x - center_x
            y_error = hand_y - center_y
            
            # Calculate target servo positions
            target_pan_pw = SERVO_MID_PW - x_error * 5
            target_tilt_pw = SERVO_MID_PW + y_error * 5
            
            # Move servos smoothly
            current_pan_pw = move_servo_smoothly(SERVO_PAN_PIN, current_pan_pw, target_pan_pw)
            current_tilt_pw = move_servo_smoothly(SERVO_TILT_PIN, current_tilt_pw, target_tilt_pw)
            
            print(f"Tracking hand at coordinates: X={hand_x}, Y={hand_y} | Servo PAN: {current_pan_pw}, TILT: {current_tilt_pw}")
            
            person_detected = True
            person_visible = True
            break  # Track only the first hand detected

    if not person_detected:
        if person_visible:  # Only print once when person disappears
            print("No human detected in frame")
            person_visible = False

    return person_detected

def record_and_process():
    global camera_connected, recording
    
    # Handle existing video file
    if os.path.exists(VIDEO_FILE):
        os.remove(VIDEO_FILE)
        print("Existing Video.mp4 deleted.")

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Check connection or try different CAMERA_ID.")
        camera_connected = False
        return
    
    camera_connected = True
    print("Camera connected successfully!")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_FILE, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
    recording = True
    
    print("Recording and tracking started. Press 'Q' to stop.")

    try:
        no_person_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame. Camera may be disconnected.")
                camera_connected = False
                break

            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Track person and update servos
            person_detected = track_person(frame)
            
            # Display status
            status = "Human Detected" if person_detected else "No Human"
            cv2.putText(frame, status, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if person_detected else (0, 0, 255), 2)
            
            # Draw center crosshair
            cv2.line(frame, (FRAME_WIDTH//2, 0), (FRAME_WIDTH//2, FRAME_HEIGHT), (255, 0, 0), 1)
            cv2.line(frame, (0, FRAME_HEIGHT//2), (FRAME_WIDTH, FRAME_HEIGHT//2), (255, 0, 0), 1)

            # Write frame to video
            out.write(frame)
            
            # Display frame
            cv2.imshow("AI Subject Tracking", frame)

            # Check for stop key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Recording stopped by user.")
                break

    except KeyboardInterrupt:
        print("Recording interrupted.")

    finally:
        recording = False
        print("Saving video file...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Run mocap.py if it exists
        if os.path.exists("mocap.py"):
            print("Running mocap.py for motion capture processing...")
            subprocess.run(["python3", "mocap.py"])
        else:
            print("WARNING: mocap.py not found in current directory")

def center_servos():
    global current_pan_pw, current_tilt_pw
    print("Centering servos...")
    pi.set_servo_pulsewidth(SERVO_PAN_PIN, SERVO_MID_PW)
    pi.set_servo_pulsewidth(SERVO_TILT_PIN, SERVO_MID_PW)
    current_pan_pw = SERVO_MID_PW
    current_tilt_pw = SERVO_MID_PW
    time.sleep(0.5)

def cleanup():
    print("Cleaning up resources...")
    pi.set_servo_pulsewidth(SERVO_PAN_PIN, 0)
    pi.set_servo_pulsewidth(SERVO_TILT_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()

def main():
    print("Starting AI Subject Tracking System...")
    print("Initializing hardware...")
    
    try:
        # Center servos at startup
        center_servos()
        
        # Start recording and tracking
        record_and_process()
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
    
    finally:
        cleanup()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
