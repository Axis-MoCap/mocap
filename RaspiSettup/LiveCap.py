#live capture script 
import os
import cv2
import mediapipe as mp
import threading
import json
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify, send_from_directory
import socket
import webbrowser
from Tracking import TrackingSystem

# Global variables
keypoints_3d = []
connections = []
is_capturing = False
_capture_thread = None
_tracking_instance = None

# MediaPipe connections for visualization
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm and hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm and hand
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg and foot
    (24, 26), (26, 28), (28, 30), (28, 32)   # Right leg and foot
]

# Flask app
app = Flask(__name__)

# Create templates and static directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# Create HTML template file
with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>3D Keypoints Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <style>
        body { margin: 0; overflow: hidden; }
        canvas { display: block; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
        }
    </style>
</head>
<body>
    <div id="info">
        <h2>3D Keypoints Visualization</h2>
        <p>Points detected: <span id="pointCount">0</span></p>
        <p>Distance: <span id="distance">0</span> cm</p>
        <p>FPS: <span id="fps">0</span></p>
    </div>
    <script>
        // Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x222222);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 2;
        
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Add orbit controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        // Add a grid helper
        const gridHelper = new THREE.GridHelper(2, 20);
        scene.add(gridHelper);
        
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 1, 0);
        scene.add(directionalLight);
        
        // Keypoints and skeleton
        let points = [];
        let lines = [];
        const pointMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00 });
        
        // Performance tracking
        let frameCount = 0;
        let lastTime = performance.now();
        
        // Animation function
        function animate() {
            requestAnimationFrame(animate);
            
            // Update FPS counter
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = Math.round(frameCount * 1000 / (now - lastTime));
                frameCount = 0;
                lastTime = now;
            }
            
            controls.update();
            renderer.render(scene, camera);
        }
        
        // Start animation
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Function to update keypoints from server
        function updateKeypoints() {
            fetch('/keypoints')
                .then(response => response.json())
                .then(data => {
                    // Update info display
                    document.getElementById('pointCount').textContent = data.keypoints.length;
                    document.getElementById('distance').textContent = data.distance;
                    
                    // Remove previous points and lines
                    points.forEach(point => scene.remove(point));
                    lines.forEach(line => scene.remove(line));
                    points = [];
                    lines = [];
                    
                    // Skip if no keypoints
                    if (data.keypoints.length === 0) return;
                    
                    // Add new points
                    data.keypoints.forEach(point => {
                        const geometry = new THREE.SphereGeometry(0.02, 8, 8);
                        const mesh = new THREE.Mesh(geometry, pointMaterial);
                        mesh.position.set(point.x, point.y, point.z);
                        scene.add(mesh);
                        points.push(mesh);
                    });
                    
                    // Add skeleton lines
                    data.connections.forEach(conn => {
                        if (conn[0] < data.keypoints.length && conn[1] < data.keypoints.length) {
                            const p1 = data.keypoints[conn[0]];
                            const p2 = data.keypoints[conn[1]];
                            
                            const geometry = new THREE.BufferGeometry().setFromPoints([
                                new THREE.Vector3(p1.x, p1.y, p1.z),
                                new THREE.Vector3(p2.x, p2.y, p2.z)
                            ]);
                            
                            const line = new THREE.Line(geometry, lineMaterial);
                            scene.add(line);
                            lines.push(line);
                        }
                    });
                })
                .catch(error => console.error('Error fetching keypoints:', error));
        }
        
        // Update keypoints regularly
        setInterval(updateKeypoints, 50); // 20 FPS update rate
    </script>
</body>
</html>
    """)

# Create offline fallback for Three.js
with open(os.path.join(os.path.dirname(__file__), 'static', 'three.min.js'), 'w') as f:
    f.write("// This is a placeholder. The full Three.js library will be downloaded at first run.\n")
    f.write("// If you're running this offline from the start, please download Three.js manually.\n")

def download_threejs_if_needed():
    """Download Three.js files if they don't exist"""
    try:
        import requests
        
        threejs_path = os.path.join(os.path.dirname(__file__), 'static', 'three.min.js')
        orbit_controls_path = os.path.join(os.path.dirname(__file__), 'static', 'OrbitControls.js')
        
        if not os.path.exists(threejs_path) or os.path.getsize(threejs_path) < 10000:
            print("Downloading Three.js library...")
            r = requests.get('https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js')
            with open(threejs_path, 'wb') as f:
                f.write(r.content)
                
        if not os.path.exists(orbit_controls_path):
            print("Downloading OrbitControls.js...")
            r = requests.get('https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js')
            with open(orbit_controls_path, 'wb') as f:
                f.write(r.content)
                
        print("Three.js files downloaded successfully.")
    except:
        print("Could not download Three.js. Will use CDN if online, or local files if offline.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/keypoints')
def get_keypoints():
    global keypoints_3d
    return jsonify({
        'keypoints': keypoints_3d,
        'connections': connections,
        'distance': _tracking_instance.Estimate_Distance if _tracking_instance else 0
    })

def capture_worker():
    """Worker function to capture and process frames"""
    global keypoints_3d, connections, is_capturing, _tracking_instance
    
    # Initialize tracking system
    _tracking_instance = TrackingSystem()
    
    try:
        # Setup camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            is_capturing = False
            return
            
        while is_capturing:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame")
                break
                
            # Track person and update keypoints
            _tracking_instance.track_person(frame)
            
            # Convert mediapipe keypoints to 3D format for visualization
            keypoints_3d = []
            if _tracking_instance.pose_kpts2d is not None:
                # Get 2D keypoints
                points_2d = _tracking_instance.pose_kpts2d
                
                # Create normalized 3D points
                # This is a simplification - for real 3D we would need depth info
                for i, point in enumerate(points_2d):
                    if i < len(points_2d):
                        # Normalize coordinates to (-1, 1) range for Three.js
                        x = (point[0] / 640) * 2 - 1
                        # Flip y-axis for Three.js coordinate system
                        y = -((point[1] / 480) * 2 - 1)
                        # Use distance estimate for z (simplified)
                        z = -_tracking_instance.Estimate_Distance / 500
                        
                        keypoints_3d.append({
                            'x': float(x),
                            'y': float(y),
                            'z': float(z)
                        })
                
                # Set up connections for the skeleton
                connections = POSE_CONNECTIONS
                
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        print(f"ERROR in capture worker: {str(e)}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if _tracking_instance is not None:
            _tracking_instance.cleanup()

def start_livecap(browser=True):
    """
    Start the 3D keypoints visualization web app
    
    Args:
        browser (bool): Whether to open browser automatically
        
    Returns:
        str: URL to access the visualization
    """
    global is_capturing, _capture_thread
    
    if is_capturing:
        return "Already running"
    
    # Try to download Three.js for offline use
    download_threejs_if_needed()
    
    # Start the capture thread
    is_capturing = True
    _capture_thread = threading.Thread(target=capture_worker)
    _capture_thread.daemon = True
    _capture_thread.start()
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Get local IP address
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        pass
    
    url = f"http://{local_ip}:8080"
    
    # Open browser if requested
    if browser:
        # Give the server a moment to start
        time.sleep(1)
        webbrowser.open(url)
    
    print(f"LiveCap is running at: {url}")
    return url

def stop_livecap():
    """Stop the 3D keypoints visualization"""
    global is_capturing, _capture_thread, _tracking_instance
    
    is_capturing = False
    
    if _capture_thread is not None:
        _capture_thread.join(timeout=2.0)
    
    if _tracking_instance is not None:
        _tracking_instance.cleanup()
        _tracking_instance = None
    
    return "LiveCap stopped"

# For testing directly
if __name__ == "__main__":
    url = start_livecap(True)
    print(f"LiveCap started at {url}")
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping LiveCap...")
        stop_livecap()