#!/usr/bin/env python3
"""
Example client for sending TIFF arrays and metadata to the autonomous microscope system.
This demonstrates how to integrate with the system from your Python session.
"""

import zmq
import numpy as np
import json
import time
from pathlib import Path

# Configuration
ANALYSIS_PORT = 5556
CONTROL_PORT = 5557

def create_sample_tiff_array(shape=(100, 100)):
    """Create a sample TIFF array for testing"""
    # Create a sample 2D array with some features
    array = np.random.normal(100, 20, shape)
    
    # Add some high-intensity regions (features of interest)
    array[20:30, 25:35] += 200  # Feature 1
    array[60:70, 45:55] += 150  # Feature 2
    array[40:50, 70:80] += 180  # Feature 3
    
    return array

def create_scan_positions(x_range=(0, 10), y_range=(0, 10), shape=(100, 100)):
    """Create scan position arrays based on scan range and array shape"""
    x_start, x_end = x_range
    y_start, y_end = y_range
    
    # Create position arrays
    x_positions = np.linspace(x_start, x_end, shape[1])
    y_positions = np.linspace(y_start, y_end, shape[0])
    
    return {
        "x": x_positions.tolist(),
        "y": y_positions.tolist()
    }

def send_tiff_array_for_analysis(tiff_array, metadata, port=ANALYSIS_PORT):
    """Send TIFF array and metadata to analysis service"""
    
    # Setup ZMQ connection
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    
    try:
        # Prepare the analysis request
        request = {
            "type": "analyze_tiff_array",
            "tiff_array": tiff_array.tolist(),  # Convert numpy array to list for JSON
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        print(f"[CLIENT] Sending TIFF array for analysis (shape: {tiff_array.shape})")
        print(f"[CLIENT] Metadata keys: {list(metadata.keys())}")
        
        # Send the request
        socket.send_json(request)
        
        # Wait for response
        response = socket.recv_json()
        print(f"[CLIENT] Analysis response received")
        print(f"[CLIENT] Found {len(response.get('regions', []))} regions")
        
        return response
        
    except Exception as e:
        print(f"[CLIENT] Error sending TIFF data for analysis: {e}")
        return None
    finally:
        socket.close()

def example_coarse_scan():
    """Example of a complete coarse scan workflow"""
    
    print("=== Autonomous Microscope Coarse Scan Example ===")
    
    # 1. Create sample TIFF array (this would be your actual scan data)
    print("\n1. Creating sample TIFF array...")
    tiff_array = create_sample_tiff_array(shape=(50, 50))
    print(f"   Array shape: {tiff_array.shape}")
    print(f"   Array range: {tiff_array.min():.1f} to {tiff_array.max():.1f}")
    
    # 2. Create metadata with scan positions
    print("\n2. Creating scan metadata...")
    metadata = {
        "scan_id": 12345,
        "scan_type": "coarse",
        "timestamp": time.time(),
        "scan_positions": create_scan_positions(
            x_range=(0, 10), 
            y_range=(0, 10), 
            shape=tiff_array.shape
        ),
        "scan_parameters": {
            "x_range": [0, 10],
            "y_range": [0, 10],
            "step_size": 0.2,
            "exposure_time": 0.1
        },
        "detector_info": {
            "name": "XRF_detector",
            "gain": 1.0,
            "integration_time": 0.1
        }
    }
    
    print(f"   Scan ID: {metadata['scan_id']}")
    print(f"   X positions: {len(metadata['scan_positions']['x'])} points")
    print(f"   Y positions: {len(metadata['scan_positions']['y'])} points")
    
    # 3. Send for analysis
    print("\n3. Sending for analysis...")
    analysis_result = send_tiff_array_for_analysis(tiff_array, metadata)
    
    if analysis_result and analysis_result.get("regions"):
        regions = analysis_result["regions"]
        print(f"\n4. Analysis Results:")
        print(f"   Found {len(regions)} regions of interest:")
        
        for i, region in enumerate(regions):
            print(f"   Region {i+1}:")
            print(f"     Position: ({region['x']:.2f}, {region['y']:.2f})")
            print(f"     Size: {region['w']:.2f} x {region['h']:.2f}")
            print(f"     Confidence: {region['confidence']:.2f}")
            if 'intensity' in region:
                print(f"     Intensity: {region['intensity']:.1f}")
            if 'array_x' in region and 'array_y' in region:
                print(f"     Array position: ({region['array_x']}, {region['array_y']})")
    
    else:
        print("\n4. Analysis failed or no regions found")
    
    return analysis_result

def example_with_real_data():
    """Example showing how to use with real scan data"""
    
    print("\n=== Real Data Integration Example ===")
    
    # This is how you would integrate with your actual scan system
    # Replace this with your actual scan code
    
    # Example: After completing a coarse scan
    def your_coarse_scan_function():
        """Your actual coarse scan function"""
        # This would be your real scan code
        # For example:
        # - Move motors to scan positions
        # - Collect detector data
        # - Return the data array
        
        print("   [SIMULATION] Running coarse scan...")
        time.sleep(2)  # Simulate scan time
        
        # Return your actual scan data
        return create_sample_tiff_array(shape=(80, 60))
    
    def your_scan_metadata_function():
        """Your function to get scan metadata"""
        # This would return your actual scan metadata
        return {
            "scan_id": int(time.time()),  # Or get from EPICS
            "scan_type": "coarse",
            "timestamp": time.time(),
            "scan_positions": {
                "x": np.linspace(0, 15, 80).tolist(),
                "y": np.linspace(0, 12, 60).tolist()
            },
            "scan_parameters": {
                "x_range": [0, 15],
                "y_range": [0, 12],
                "step_size": 0.19,
                "exposure_time": 0.1
            }
        }
    
    # Run the scan
    print("1. Running coarse scan...")
    tiff_array = your_coarse_scan_function()
    metadata = your_scan_metadata_function()
    
    print(f"   Scan completed: {tiff_array.shape}")
    
    # Send for analysis
    print("2. Sending for analysis...")
    result = send_tiff_array_for_analysis(tiff_array, metadata)
    
    if result:
        print("3. Analysis completed successfully")
        return result
    else:
        print("3. Analysis failed")
        return None

if __name__ == "__main__":
    # Run the example
    example_coarse_scan()
    
    # Uncomment to see real data integration example
    # example_with_real_data()
    
    print("\n=== Integration Notes ===")
    print("To integrate with your system:")
    print("1. Replace create_sample_tiff_array() with your actual scan data")
    print("2. Replace create_scan_positions() with your actual motor positions")
    print("3. Call send_tiff_array_for_analysis() after each coarse scan")
    print("4. The system will automatically generate zoom scan plans")
    print("5. Zoom scans will be submitted to your Bluesky queue") 