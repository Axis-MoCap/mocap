# MoCap

Firstly be sure that you are using a 64bit Raspberry pi OS ⚠️32 bit may throw errors (especially with mediapipe)⚠️

✅ How the MoCap System Works – Step-by-Step Guide
1. Preparing Your Custom Character Model (On a PC)
If you want to use your own custom character:

Open Blender and import your .blend file containing the character.

Make sure your character has a suitable armature (skeleton). The bones should be named appropriately — following the structure used in the sample file found in the ASSETS folder (maximo.blend).

Once your armature is ready:

Switch to the Scripting tab in Blender.

Run the export_skeleton.py script.

This will generate a tmp folder with a subfolder called Skeleton, which contains:

bone_matrix.npy:
→ Stores the local transformation matrices of each bone — this defines how each bone is positioned relative to its parent.

bone_matrix_world.npy:
→ Stores the global/world-space matrices — defining the absolute position of each bone in 3D space.

skeleton.json:
→ A full export of your armature's structure — including bone names, hierarchies, and parent-child relationships. This file is used to match the animation data to your skeleton later.

2. Running the MoCap System (On Raspberry Pi or VM)
To extract motion data:

Run the main script mocap.py.

This will:

Load the video file (e.g. TestVideo.mp4).

Use MediaPipe for pose detection to track body landmarks in each video frame.

Use Skeleton IK to convert those 2D body points into 3D bone transformations that fit your custom skeleton.

Export the results as a .pkl (Pickle) file, which contains:

A frame-by-frame list of bone positions and rotations

Metadata like frame rate and the structure of the tracked skeleton

This .pkl file is saved in the tmp folder and will be used to apply animation to your character.

3. Applying the Animation (Back in Blender on Your PC)
To bring your character to life:

Open Blender and create a new project.

Load the .blend file of your custom character.

Go to the Scripting tab and run apply_animation.py.

This script:

Loads the .pkl file you created with mocap.py

Matches the motion data to your character’s skeleton (based on the exported skeleton.json)

Applies the full animation — you’ll see your character move exactly as captured from the video!

⚠️I WILL 100% WORK ON MAKING THIS MORE STREAMLINED AND EASY BUT FOR NOW GOODLUCK⚠️


✅ Testing the MoCap System on Raspberry Pi – Demo Guide
1. Install the Required Dependencies
Open the Dependencies.txt file from the GitHub repository.

Copy and paste each command into your Raspberry Pi terminal to install all the necessary libraries.

Make sure:

-Python 3 is installed

-You have internet access during the installation

-You’re using the correct pip version (python3 -m pip install ... if needed)

2. Run the Motion Capture Script
Before running:

Make sure the folder structure on your Raspberry Pi matches exactly how it is laid out on the GitHub repository. This includes the mocap.py script, assets, and the supporting scripts and folders.

Once it runs:

It will begin processing the video file (e.g., TestVideo.mp4)

A loading bar will appear in the terminal showing progress

⏳ Processing may take up to 1 hour, depending on your Pi model (due to optimization limits).
🔴 Do not interrupt or close the terminal while it runs

3. Check if the Processing Was Successful
After processing:

Navigate to the tmp folder (automatically created by the script)

Confirm that a file named: bone_animation_data.pkl  has been generated.

This file contains the captured motion data and is needed for animation.

4. (Optional) Preview the Animation on Your PC Using Blender
Once your Raspberry Pi has finished processing the video and created the bone_animation_data.pkl file, you can test the results visually on your PC.

Here's how to do it:
✅ Step-by-step:

Transfer the files from your Raspberry Pi to your PC

Copy the entire RaspiSettup folder from the Raspberry Pi.

This ensures you bring along all necessary scripts, models, and the tmp folder (which contains the .pkl animation file).

Open Blender on your PC

Start a new Blender project.

Go to the Scripting tab at the top of the Blender window.

Run the animation script

Load and run the apply_animation.py script inside Blender.

What happens next:

Blender will import your character model.

It will apply the motion data from bone_animation_data.pkl.

You should now see your character moving as captured from the video!

🎥 End Result:
You’ve successfully taken motion capture data from a Raspberry Pi and visualized it in Blender on your PC — no special motion-tracking hardware needed.

Let me know if you want this turned into a PDF guide or README format!
