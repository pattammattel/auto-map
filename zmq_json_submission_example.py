#!/usr/bin/env python3
"""
Example script demonstrating ZMQ JSON path submission for queue management.
This shows how to send JSON file paths over ZMQ instead of relying on file system discovery.
"""

import sys
import os
import json
import zmq
import time
import numpy as np
from pathlib import Path

# Configuration
ANALYSIS_PORT = 5556
SUBMIT_DIR = Path("./submitted")

def create_sample_json_file():
    """Create a sample zoom boxes JSON file for testing"""
    zoom_boxes = {
        "zoom_scan_123_region_1": {
            "real_center_um": [1000.0, 500.0],
            "real_size_um": [100.0, 100.0],
            "confidence": 0.85,
            "array_position": {"x": 10, "y": 5},
            "intensity": 1500.0,
            "timestamp": time.time()
        },
        "zoom_scan_123_region_2": {
            "real_center_um": [1500.0, 800.0],
            "real_size_um": [50.0, 50.0],
            "confidence": 0.92,
            "array_position": {"x": 15, "y": 8},
            "intensity": 2200.0,
            "timestamp": time.time()
        }
    }
    
    # Create submit directory if it doesn't exist
    SUBMIT_DIR.mkdir(exist_ok=True)
    
    # Save JSON file
    json_filename = f"zoom_boxes_{int(time.time())}.json"
    json_path = SUBMIT_DIR / json_filename
    
    with open(json_path, "w") as f:
        json.dump(zoom_boxes, f, indent=2)
    
    print(f"[EXAMPLE] Created sample JSON file: {json_path}")
    return json_path

def send_json_path_via_zmq(json_path, metadata=None):
    """Send JSON file path to analysis service via ZMQ"""
    
    # Setup ZMQ connection
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{ANALYSIS_PORT}")
    
    try:
        # Prepare the queue submission request
        request = {
            "type": "submit_json_boxes",
            "json_path": str(json_path),
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        print(f"[EXAMPLE] Sending JSON path for queue submission: {json_path}")
        print(f"[EXAMPLE] Metadata: {metadata}")
        
        # Send the request
        socket.send_json(request)
        
        # Wait for response
        response = socket.recv_json()
        print(f"[EXAMPLE] Queue submission response received: {response}")
        
        return response
        
    except Exception as e:
        print(f"[EXAMPLE] Error sending JSON path for queue submission: {e}")
        return None
    finally:
        socket.close()

def main():
    """Main function demonstrating ZMQ JSON path submission"""
    print(f"[EXAMPLE] ZMQ JSON Path Submission Example")
    print(f"[EXAMPLE] Analysis port: {ANALYSIS_PORT}")
    print(f"[EXAMPLE] Submit directory: {SUBMIT_DIR}")
    
    # Create a sample JSON file
    json_path = create_sample_json_file()
    
    # Prepare metadata
    metadata = {
        "scan_id": 123,
        "zoom_plans_count": 2,
        "source": "example_script",
        "timestamp": time.time()
    }
    
    # Send JSON path via ZMQ
    print(f"\n[EXAMPLE] Sending JSON path via ZMQ...")
    response = send_json_path_via_zmq(json_path, metadata)
    
    if response:
        if response.get("success"):
            print(f"[EXAMPLE] ✅ Successfully submitted JSON boxes to queue")
        else:
            error = response.get("error", "Unknown error")
            print(f"[EXAMPLE] ❌ Failed to submit JSON boxes: {error}")
    else:
        print(f"[EXAMPLE] ❌ No response received from analysis service")
    
    print(f"\n[EXAMPLE] Example completed")

if __name__ == "__main__":
    main() 