import requests
import time
import json

def print_response(response):
    print(json.dumps(response.json(), indent=2))

# Base URL - use localhost for testing
BASE_URL = "http://localhost:5000"

# Start the test
print("Testing Flask API for motion tracking system")
print("--------------------------------------------")

# 1. Check status
print("\n1. Checking status...")
response = requests.get(f"{BASE_URL}/status")
print_response(response)

# 2. Center servos
print("\n2. Centering servos...")
response = requests.post(f"{BASE_URL}/center_servos")
print_response(response)

# 3. Start tracking in test mode
print("\n3. Starting tracking...")
response = requests.post(f"{BASE_URL}/start_tracking")
print_response(response)

# 4. Poll status a few times
print("\n4. Polling status (3 times)...")
for i in range(3):
    time.sleep(1)
    response = requests.get(f"{BASE_URL}/status")
    print_response(response)

# 5. Set servo position
print("\n5. Setting servo position...")
data = {"pan": 1200, "tilt": 1800}
response = requests.post(f"{BASE_URL}/servo_position", json=data)
print_response(response)

# 6. Get servo position
print("\n6. Getting servo position...")
response = requests.get(f"{BASE_URL}/servo_position")
print_response(response)

# 7. Stop tracking
print("\n7. Stopping tracking...")
response = requests.post(f"{BASE_URL}/stop_tracking")
print_response(response)

# 8. Run mocap processing
print("\n8. Running mocap processing...")
response = requests.post(f"{BASE_URL}/run_mocap")
print_response(response)

# 9. Cleanup
print("\n9. Cleaning up resources...")
response = requests.post(f"{BASE_URL}/cleanup")
print_response(response)

print("\nTest completed!")