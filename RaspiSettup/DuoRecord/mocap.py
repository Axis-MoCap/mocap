import os
import shutil
import argparse
import pickle
import subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm
from body_keypoint_track import BodyKeypointTrack, show_annotation
from skeleton_ik_solver import SkeletonIKSolver

def main():
    # Check for Video.mp4 first
    video_path = 'Video.mp4'
    
    # If Video.mp4 doesn't exist, run Tracking.py
    if not os.path.exists(video_path):
        print(f"'{video_path}' not found. Running Tracking.py...")
        tracking_script = "Tracking.py"
        if not os.path.exists(tracking_script):
            raise Exception(f"'{tracking_script}' not found in the current directory.")
        
        # Run the Tracking.py script
        proc = subprocess.Popen(f"python {tracking_script}")
        proc.wait()
        
        # Check again if Video.mp4 was created by the tracking script
        if not os.path.exists(video_path):
            raise Exception(f"'{video_path}' was not created by the tracking script.")
    
    # Path to the blender model
    blend_path = 'assets/skeleton.blend'  # Your rigged model .blend file
    FOV = np.pi / 3  # Field of view, set to 60 degrees
    
    # Ensure the tmp folder exists
    os.makedirs('tmp', exist_ok=True)
    
    # Step 1: Ensure that the skeleton folder is already present


#From here we need ro first make it display two seperate frames for each individual and track each of their movements






    
    if not os.path.exists('tmp/skeleton'):
        raise Exception("Skeleton export failed. Please ensure the skeleton is exported and placed in 'tmp/skeleton'.")
    
    # Step 2: Open the video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Video capture failed for '{video_path}'")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize body keypoint tracker
    body_keypoint_track = BodyKeypointTrack(
        im_width=frame_width,
        im_height=frame_height,
        fov=FOV,
        frame_rate=frame_rate,
        track_hands=True,
        smooth_range=10 * (1 / frame_rate),
        smooth_range_barycenter=30 * (1 / frame_rate),
    )
    
    # Initialize the skeleton IK solver
    skeleton_ik_solver = SkeletonIKSolver(
        model_path='tmp/skeleton',
        track_hands=False,
        smooth_range=15 * (1 / frame_rate),
    )
    
    # Data lists to store bone data
    bone_euler_sequence, scale_sequence, location_sequence = [], [], []
    
    # Time tracking
    frame_t = 0.0
    frame_i = 0
    bar = tqdm(total=total_frames, desc='Running...')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get 3D body keypoints
        body_keypoint_track.track(frame, frame_t)
        kpts3d, valid = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)
        
        # Solve for the skeleton pose using IK
        skeleton_ik_solver.fit(torch.from_numpy(kpts3d).float(), torch.from_numpy(valid).bool(), frame_t)
        
        # Get smoothed pose data
        bone_euler = skeleton_ik_solver.get_smoothed_bone_euler(frame_t)
        location = skeleton_ik_solver.get_smoothed_location(frame_t)
        scale = skeleton_ik_solver.get_scale()
        
        # Append the data to the sequences
        bone_euler_sequence.append(bone_euler)
        location_sequence.append(location)
        scale_sequence.append(scale)
        
        # Show the keypoints on the frame (optional)
        show_annotation(frame, kpts3d, valid, body_keypoint_track.K)
        
        if cv2.waitKey(1) == 27:  # Exit if 'ESC' is pressed
            print('Cancelled by user. Exit.')
            exit()
        
        # Increment frame time
        frame_i += 1
        frame_t += 1.0 / frame_rate
        bar.update(1)
    
    # Step 3: Save animation result as a pickle file
    print("Save animation result...")
    with open('tmp/bone_animation_data.pkl', 'wb') as fp:
        pickle.dump({
            'fov': FOV,
            'frame_rate': frame_rate,
            'bone_names': skeleton_ik_solver.optimizable_bones,
            'bone_euler_sequence': bone_euler_sequence,
            'location_sequence': location_sequence,
            'scale': np.mean(scale_sequence),
            'all_bone_names': skeleton_ik_solver.all_bone_names
        }, fp)
    
    # Step 4: Open Blender and apply the animation to the rigged model
    print("Open blender and apply animation...")
    proc = subprocess.Popen(f"blender {blend_path} --python apply_animation.py")
    proc.wait()

if __name__ == '__main__':
    main()
