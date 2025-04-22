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

# Config
SERVO_PAN_PIN = 18
SERVO_TILT_PIN = 19
CAMERA_ID = 0
SERVO_MIN_PW = 500
SERVO_MAX_PW = 2500
SERVO_MID_PW = 1500
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SMOOTHING_FACTOR = 0.3
VIDEO_FILE = "Video.mp4"

class TrackingSystem:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise Exception("ERROR: pigpio connection failed")
            
        self.current_pan_pw = SERVO_MID_PW
        self.current_tilt_pw = SERVO_MID_PW
        self.camera_connected = False
        self.person_visible = False
        self.recording = False
        self.stop_requested = False
        self.Estimate_Distance = 0  # Initialize distance estimate
        self.visible_keypoints_count = 0  # Track number of visible keypoints
        
        # Initialize servo positions
        self.pi.set_servo_pulsewidth(SERVO_PAN_PIN, self.current_pan_pw)
        self.pi.set_servo_pulsewidth(SERVO_TILT_PIN, self.current_tilt_pw)

    def map_value(self, value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def move_servo_smoothly(self, servo_pin, current_pw, target_pw):
        new_pw = current_pw + SMOOTHING_FACTOR * (target_pw - current_pw)
        new_pw = max(SERVO_MIN_PW, min(SERVO_MAX_PW, new_pw))
        self.pi.set_servo_pulsewidth(servo_pin, new_pw)
        return new_pw

    def calculate_distance(self, landmarks, frame_width, frame_height):
        """Estimate distance based on the size of the person in the frame"""
        if not landmarks:
            return 0
            
        # Get bounding box of visible landmarks
        visible_landmarks = [lmk for lmk in landmarks if lmk.visibility > 0.5]
        self.visible_keypoints_count = len(visible_landmarks)  # Update visible keypoints count
        
        if not visible_landmarks:
            return 0
            
        xs = [lmk.x * frame_width for lmk in visible_landmarks]
        ys = [lmk.y * frame_height for lmk in visible_landmarks]
        
        # Calculate bounding box dimensions
        if not xs or not ys:
            return 0
            
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # The bigger the person appears in frame, the closer they are
        # This is a very simple heuristic - can be calibrated for better accuracy
        size_factor = width * height / (frame_width * frame_height)
        
        # Map size_factor to an estimated distance in cm
        # These values would need calibration for specific camera setup
        if size_factor > 0:
            distance = self.map_value(size_factor, 0.01, 0.5, 300, 50)
            return int(distance)
        return 0

    def draw_bounding_box(self, frame, landmarks):
        """Draw a bounding box around the person"""
        h, w, _ = frame.shape
        
        # Filter for visible landmarks
        visible_landmarks = [lmk for lmk in landmarks if lmk.visibility > 0.5]
        
        if not visible_landmarks:
            return frame
            
        # Get x, y coordinates for visible landmarks
        xs = [lmk.x * w for lmk in visible_landmarks]
        ys = [lmk.y * h for lmk in visible_landmarks]
        
        # Calculate bounding box
        if xs and ys:
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            
            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
        return frame

    def track_person(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        hands_results = hands.process(frame_rgb)

        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2
        person_detected = False

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw bounding box around person
            frame = self.draw_bounding_box(frame, pose_results.pose_landmarks.landmark)
            
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            
            # Calculate distance estimate
            self.Estimate_Distance = self.calculate_distance(pose_results.pose_landmarks.landmark, w, h)
            
            x_error = nose_x - center_x
            y_error = nose_y - center_y
            
            target_pan_pw = SERVO_MID_PW - x_error * 5
            target_tilt_pw = SERVO_MID_PW + y_error * 5
            
            self.current_pan_pw = self.move_servo_smoothly(SERVO_PAN_PIN, self.current_pan_pw, target_pan_pw)
            self.current_tilt_pw = self.move_servo_smoothly(SERVO_TILT_PIN, self.current_tilt_pw, target_tilt_pw)
            
            cv2.line(frame, (nose_x, 30), (nose_x, 60), (0, 255, 0), 2)
            cv2.line(frame, (30, nose_y), (60, nose_y), (0, 255, 0), 2)
            
            person_detected = True
            self.person_visible = True

        elif hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw bounding box around hand
                hand_landmarks_list = list(hand_landmarks.landmark)
                frame = self.draw_bounding_box(frame, hand_landmarks_list)
                
                xs = [lmk.x * w for lmk in hand_landmarks.landmark]
                ys = [lmk.y * h for lmk in hand_landmarks.landmark]
                hand_x = int(sum(xs) / len(xs))
                hand_y = int(sum(ys) / len(ys))
                
                # Calculate distance estimate for hands
                hand_landmarks_list = list(hand_landmarks.landmark)
                self.Estimate_Distance = self.calculate_distance(hand_landmarks_list, w, h)
                
                x_error = hand_x - center_x
                y_error = hand_y - center_y
                
                target_pan_pw = SERVO_MID_PW - x_error * 5
                target_tilt_pw = SERVO_MID_PW + y_error * 5
                
                self.current_pan_pw = self.move_servo_smoothly(SERVO_PAN_PIN, self.current_pan_pw, target_pan_pw)
                self.current_tilt_pw = self.move_servo_smoothly(SERVO_TILT_PIN, self.current_tilt_pw, target_tilt_pw)
                
                person_detected = True
                self.person_visible = True
                break

        if not person_detected and self.person_visible:
            self.person_visible = False
            self.Estimate_Distance = 0
            self.visible_keypoints_count = 0

        return person_detected

    def start_tracking(self):
        """Start the tracking and recording process"""
        self.stop_requested = False
        
        # Delete existing video file if it exists
        if os.path.exists(VIDEO_FILE):
            os.remove(VIDEO_FILE)
            print("Existing Video.mp4 deleted.")

        # Initialize camera
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            raise Exception("ERROR: Could not open camera")

        self.camera_connected = True
        print("Camera connected successfully!")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(VIDEO_FILE, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
        self.recording = True

        try:
            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame")

                # Add timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Track person and update servos
                person_detected = self.track_person(frame)
                
                # Add status overlay
                status = "Human Detected" if person_detected else "No Human"
                cv2.putText(frame, status, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 255, 0) if person_detected else (0, 0, 255), 2)
                
                # Draw center crosshair
                cv2.line(frame, (FRAME_WIDTH//2, 0), (FRAME_WIDTH//2, FRAME_HEIGHT), (255, 0, 0), 1)
                cv2.line(frame, (0, FRAME_HEIGHT//2), (FRAME_WIDTH, FRAME_HEIGHT//2), (255, 0, 0), 1)
                
                # Display estimated distance
                distance_text = f"Estimated Distance: {self.Estimate_Distance} cm"
                cv2.putText(frame, distance_text, (10, FRAME_HEIGHT - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display visible keypoints count
                keypoints_text = f"Visible Keypoints: {self.visible_keypoints_count}"
                cv2.putText(frame, keypoints_text, (10, FRAME_HEIGHT - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Write frame to video
                out.write(frame)
                
                # Display frame
                cv2.imshow("AI Subject Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.recording = False
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Run mocap.py if it exists
            if os.path.exists("mocap.py"):
                print("Running mocap.py for motion capture processing...")
                subprocess.run(["python3", "mocap.py"])

    def stop_tracking(self):
        """Stop the tracking process"""
        self.stop_requested = True
        self.center_servos()

    def center_servos(self):
        """Center both servos"""
        print("Centering servos...")
        self.pi.set_servo_pulsewidth(SERVO_PAN_PIN, SERVO_MID_PW)
        self.pi.set_servo_pulsewidth(SERVO_TILT_PIN, SERVO_MID_PW)
        self.current_pan_pw = SERVO_MID_PW
        self.current_tilt_pw = SERVO_MID_PW
        time.sleep(0.5)

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        self.pi.set_servo_pulsewidth(SERVO_PAN_PIN, 0)
        self.pi.set_servo_pulsewidth(SERVO_TILT_PIN, 0)
        self.pi.stop()
        cv2.destroyAllWindows()

def main():
    """Main function for testing"""
    tracker = TrackingSystem()
    try:
        tracker.center_servos()
        tracker.start_tracking()
    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        tracker.cleanup()

# Function to be called from Flutter app
def start_tracking_from_flutter():
    """
    Function to be called from the Flutter app.
    Starts the camera feed with tracking, displays distance and keypoints count,
    and returns data that can be used by the Flutter app.
    """
    tracking_data = {
        "keypoints_count": 0,
        "distance": 0,
        "tracking_status": "Starting"
    }
    
    # Run in a background thread to avoid blocking the Flutter app
    def tracking_thread():
        tracker = TrackingSystem()
        try:
            tracker.center_servos()
            
            # Setup camera
            cap = cv2.VideoCapture(CAMERA_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            if not cap.isOpened():
                tracking_data["tracking_status"] = "Failed to connect camera"
                return
                
            # Delete existing video file if it exists
            if os.path.exists(VIDEO_FILE):
                os.remove(VIDEO_FILE)
                
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(VIDEO_FILE, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
            
            tracking_data["tracking_status"] = "Running"
            
            while not tracker.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    tracking_data["tracking_status"] = "Camera disconnected"
                    break
                
                # Track person and update servos
                person_detected = tracker.track_person(frame)
                
                # Update tracking data for Flutter
                tracking_data["keypoints_count"] = tracker.visible_keypoints_count
                tracking_data["distance"] = tracker.Estimate_Distance
                
                # Add visual elements to frame
                # Timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Status
                status = "Human Detected" if person_detected else "No Human"
                cv2.putText(frame, status, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 255, 0) if person_detected else (0, 0, 255), 2)
                
                # Center crosshair
                cv2.line(frame, (FRAME_WIDTH//2, 0), (FRAME_WIDTH//2, FRAME_HEIGHT), (255, 0, 0), 1)
                cv2.line(frame, (0, FRAME_HEIGHT//2), (FRAME_WIDTH, FRAME_HEIGHT//2), (255, 0, 0), 1)
                
                # Distance
                distance_text = f"Estimated Distance: {tracker.Estimate_Distance} cm"
                cv2.putText(frame, distance_text, (10, FRAME_HEIGHT - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Keypoints count
                keypoints_text = f"Visible Keypoints: {tracker.visible_keypoints_count}"
                cv2.putText(frame, keypoints_text, (10, FRAME_HEIGHT - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Write frame to video
                out.write(frame)
                
                # Display frame
                cv2.imshow("AI Tracking (Flutter)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            tracking_data["tracking_status"] = "Stopped"
            
            # Save recording
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Run mocap.py
            if os.path.exists("mocap.py"):
                subprocess.run(["python3", "mocap.py"])
            
        except Exception as e:
            tracking_data["tracking_status"] = f"Error: {str(e)}"
        finally:
            tracker.cleanup()
    
    # Start tracking in background thread
    thread = threading.Thread(target=tracking_thread)
    thread.daemon = True
    thread.start()
    
    return tracking_data

# Function to stop tracking from Flutter
def stop_tracking_from_flutter():
    """Stop the tracking process that was started from Flutter"""
    # This is a simple implementation. In practice, you would need to
    # access the tracker instance from the tracking thread.
    cv2.destroyAllWindows()
    
    # In a real implementation, you would set the stop_requested flag
    # of the tracker instance used in the tracking thread.

if __name__ == "__main__":
    main()
