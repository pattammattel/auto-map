#!/usr/bin/env python3
"""
Simplified Hardware-Connected Queue-Based Autonomous Microscope Controller
- Uses caget for scan numbers
- Integrated analysis in main session
- Custom analysis code for zoom box generation
"""

import sys
import os
import json
import time
import numpy as np
import signal
import argparse
from pathlib import Path

# Configuration
TIFF_DIR = Path("./tiff_dir")
SUBMIT_DIR = Path("./submitted")

# EPICS Configuration
DEFAULT_SCAN_PV = "SCAN:ID" #"XF:03IDC-ES{Status}ScanID-I"
DEFAULT_STATUS_PV = "SCAN:RUNNING" #"XF:03IDC-ES{Status}ScanRunning-I"

print(f"[INIT] Simplified hardware queue-based auto-map system starting...")
print(f"[INIT] TIFF directory: {TIFF_DIR}")
print(f"[INIT] Submit directory: {SUBMIT_DIR}")
print(f"[INIT] Default scan ID PV: {DEFAULT_SCAN_PV}")
print(f"[INIT] Default status PV: {DEFAULT_STATUS_PV}")


class SimpleEPICSManager:
    """Simple EPICS manager using caget for scan numbers"""
    
    def __init__(self, scan_pv_name= None, status_pv_name= None):
        self.scan_pv_name = scan_pv_name
        self.status_pv_name = status_pv_name
        self.current_scan_id = 0
        self.current_scan_status = "idle"
        self.connected = False
        
        print(f"[EPICS] Simple EPICS manager initialized with PVs: {scan_pv_name}, {status_pv_name}")
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test EPICS connection using caget"""
        try:
            import epics
            scan_id = epics.caget(self.scan_pv_name)
            status_bool = epics.caget(self.status_pv_name)
            
            if scan_id is not None and status_bool is not None:
                self.connected = True
                self.current_scan_id = scan_id
                self.current_scan_status = "running" if status_bool else "idle"
                print(f"[EPICS] Connection successful")
                print(f"[EPICS] Current scan ID: {scan_id}")
                print(f"[EPICS] Current status: {self.current_scan_status}")
                return True
            else:
                print(f"[EPICS] Connection failed - PVs returned None")
                return False
                
        except ImportError:
            print(f"[EPICS] ERROR: epics module not available. Install with: pip install pyepics")
            return False
        except Exception as e:
            print(f"[EPICS] Connection test failed: {e}")
            return False
    
    def get_current_scan_id(self):
        """Get current scan ID using caget"""
        try:
            scan_id = epics.caget(self.scan_pv_name)
            if scan_id is not None:
                self.current_scan_id = scan_id
            return self.current_scan_id
        except Exception as e:
            print(f"[EPICS] Error getting scan ID: {e}")
            return self.current_scan_id
    
    def get_scan_status(self):
        """Get current scan status using caget"""
        try:
            status_bool = epics.caget(self.status_pv_name)
            if status_bool is not None:
                if status_bool:
                    self.current_scan_status = "running"
                else:
                    # If was running and now False, it completed
                    if self.current_scan_status == "running":
                        self.current_scan_status = "completed"
                    else:
                        self.current_scan_status = "idle"
            return self.current_scan_status
        except Exception as e:
            print(f"[EPICS] Error getting scan status: {e}")
            return self.current_scan_status
    
    def wait_for_scan_completion(self, timeout=600):
        """Wait for scan to complete"""
        print(f"[EPICS] Waiting for scan completion...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_scan_status()
            if status == "completed":
                print(f"[EPICS] Scan completed")
                return True
            elif status == "error":
                print(f"[EPICS] Scan failed with error")
                return False
            time.sleep(1)
        
        print(f"[EPICS] Timeout waiting for scan completion")
        return False


class SimpleQueueManager:
    """Simple queue manager for scan submission, with persistent controls.json config."""
    
    def __init__(self, controls_json_path):
        """
        Initialize the queue manager, loading controls.json and storing its path.
        """
        self.queue_connected = False
        self.controls_json_path = controls_json_path
        self.microscope_config = {}
        self.reload_controls()

        # Try to connect to Bluesky queue server
        try:
            import bluesky_queueserver_api
            self.queue_connected = True
            print(f"[QUEUE] Connected to Bluesky queue server")
        except ImportError:
            print(f"[QUEUE] WARNING: bluesky_queueserver_api not available")
            print(f"[QUEUE] Install with: pip install bluesky-queueserver-api")
            self.queue_connected = False

    def reload_controls(self):
        """
        Reload controls.json from disk and update self.controls.
        """
        try:
            with open(self.controls_json_path, "r") as f:
                self.microscope_config = json.load(f)    
            print(f"[QUEUE] Reloaded controls from {self.controls_json_path}")
        except Exception as e:
            print(f"[QUEUE] ERROR: Could not reload controls.json: {e}")
            self.microscope_config = {}

    def submit_coarse_scan(self, coarse_scan_params_path):
        """
        Submit coarse scan to queue using provided coarse scan param JSON path.
        Uses self.controls_json_path for controls and microscope config.
        """
        try:
            if not self.queue_connected:
                print(f"[QUEUE] Cannot submit - queue server not connected")
                return False

            if not self.microscope_config:
                print(f"[QUEUE] Aborting scan submission due to missing controls.json or fields.")
                time.sleep(5)
                return False

            label = self.microscope_config.get("label", f"scan_{int(time.time())}")
            coarse_scan_params = load_coarse_scan_params(coarse_scan_params_path)
            x_motor = self.microscope_config[coarse_scan_params["mot1"]]
            y_motor = self.microscope_config[coarse_scan_params["mot2"]]
            det_name = coarse_scan_params["det_name"]
            x_start = coarse_scan_params["mot1_s"]
            x_end = coarse_scan_params["mot1_e"]
            num_x = coarse_scan_params["mot1_n"]
            y_start = coarse_scan_params["mot2_s"]
            y_end = coarse_scan_params["mot2_e"]
            num_y = coarse_scan_params["mot2_n"]
            exp_t = coarse_scan_params["exp_t"]

            roi = {x_motor: (x_start + x_end) / 2, y_motor: (y_start + y_end) / 2}

            try:
                from bluesky_queueserver_api import BPlan, REManagerAPI
                RM = REManagerAPI()
                RM.item_add(BPlan(
                    "recover_pos_and_scan",
                    label,
                    roi,
                    det_name,
                    x_motor,
                    x_start,
                    x_end,
                    num_x,
                    y_motor,
                    y_start,
                    y_end,
                    num_y,
                    exp_t
                ))
                print(f"[QUEUE] Submitted scan {label} to Bluesky queue.")
                return True
            except Exception as e:
                print(f"[QUEUE] Failed to submit scan: {e}")
                return False

        except Exception as e:
            print(f"[QUEUE] Failed to submit coarse scan: {e}")
            return False

    def submit_zoom_scans(self, json_file_path_boxes, microscope_config, fine_scan_params_path):
        """
        Submit zoom scan plans from a JSON file generated by the analysis class.
        - json_file_path: path to the zoom boxes JSON file
        - microscope_config: dict mapping logical to hardware motor names
        - fine_scan_params_path: path to the fine scan parameter JSON file
        """
        try:
            if not self.queue_connected:
                print(f"[QUEUE] Cannot submit - queue server not connected")
                return False

            from bluesky_queueserver_api import BPlan, REManagerAPI
            RM = REManagerAPI()

            fine_params = load_fine_scan_params(fine_scan_params_path)
            x_motor = self.microscope_config[fine_params["mot1"]]
            y_motor = self.microscope_config[fine_params["mot2"]]
            exp_t = fine_params["exp_t"]
            det_name = fine_params["det_name"]

            with open(json_file_path_boxes, "r") as f:
                boxes = json.load(f)

            for label, info in boxes.items():
                cx, cy = info["real_center_um"]         # center in um
                sx, sy = info["real_size_um"]           # size in um
                num_x = int(sx)
                num_y = int(sy)

                # Define relative scan range around center
                x_start = -sx / 2
                x_end = sx / 2
                y_start = -sy / 2
                y_end = sy / 2

                # Create ROI dictionary to move motors first
                roi = {x_motor: cx, y_motor: cy}

                # Submit to Bluesky queue
                RM.item_add(BPlan(
                    "recover_pos_and_scan",
                    label,
                    roi,
                    det_name,
                    x_motor,
                    x_start,
                    x_end,
                    num_x,
                    y_motor,
                    y_start,
                    y_end,
                    num_y,
                    exp_t
                ))
                print(f"[QUEUE] Queued zoom scan: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")
            return True
        except Exception as e:
            print(f"[QUEUE] Failed to submit zoom scans: {e}")
            return False


class CustomAnalysis:
    """Custom analysis class for zoom box generation"""
    
    def __init__(self):
        print(f"[ANALYSIS] Custom analysis initialized")
    
    def load_scan_data(self, scan_id):
        """Load scan data as numpy array"""
        print(f"[ANALYSIS] Loading data for scan {scan_id}")
        
        # TODO: Replace with your actual data loading code
        # Example:
        # data = load_tiff_file(f"scan_{scan_id}.tiff")
        # return np.array(data)
        
        # For now, simulate data loading
        print(f"[ANALYSIS] Simulating data loading for scan {scan_id}")
        data = np.random.rand(100, 100)  # Replace with real data loading
        return data
    
    def analyze_data(self, data, scan_id):
        """Your custom analysis code to generate zoom boxes"""
        print(f"[ANALYSIS] Analyzing data with shape {data.shape}")
        
        # TODO: Replace with your actual analysis code
        # This is where you put your custom analysis logic
        
        # Example analysis (replace with your code):
        regions = []
        
        # Simple threshold-based detection
        threshold = np.mean(data) + 2 * np.std(data)
        high_intensity_coords = np.where(data > threshold)
        
        if len(high_intensity_coords[0]) > 0:
            for i in range(min(3, len(high_intensity_coords[0]))):
                y_idx = high_intensity_coords[0][i] if i < len(high_intensity_coords[0]) else 0
                x_idx = high_intensity_coords[1][i] if i < len(high_intensity_coords[1]) else 0
                
                regions.append({
                    "x": float(x_idx),
                    "y": float(y_idx),
                    "w": 1.0,
                    "h": 1.0,
                    "confidence": 0.85 + i * 0.05,
                    "array_x": x_idx,
                    "array_y": y_idx,
                    "intensity": float(data[y_idx, x_idx])
                })
        
        # If no regions found, create default ones
        if not regions:
            regions = [
                {"x": 10.0, "y": 5.0, "w": 1.0, "h": 1.0, "confidence": 0.85},
                {"x": 15.0, "y": 8.0, "w": 0.5, "h": 0.5, "confidence": 0.92},
                {"x": 7.0, "y": 12.0, "w": 0.8, "h": 0.8, "confidence": 0.78}
            ]
        
        print(f"[ANALYSIS] Found {len(regions)} regions")
        return regions
    
    def create_zoom_boxes_json(self, regions, scan_id):
        """Create JSON file with zoom boxes in the expected format"""
        print(f"[ANALYSIS] Creating zoom boxes JSON for scan {scan_id}")
        
        # Create zoom boxes dictionary
        zoom_boxes = {}
        
        for i, region in enumerate(regions):
            # Convert scan coordinates to micrometers (adjust conversion as needed)
            um_per_mm = 1000.0
            cx_um = region["x"] * um_per_mm  # center X in um
            cy_um = region["y"] * um_per_mm  # center Y in um
            sx_um = region["w"] * um_per_mm  # size X in um
            sy_um = region["h"] * um_per_mm  # size Y in um
            
            # Create label for this region
            label = f"zoom_scan_{scan_id}_region_{i+1}"
            
            zoom_boxes[label] = {
                "real_center_um": [cx_um, cy_um],
                "real_size_um": [sx_um, sy_um],
                "confidence": region.get("confidence", 0.0),
                "array_position": {
                    "x": region.get("array_x", 0),
                    "y": region.get("array_y", 0)
                },
                "intensity": region.get("intensity", 0.0),
                "timestamp": time.time()
            }
        
        # Save JSON file
        json_filename = f"zoom_boxes_{scan_id}_{int(time.time())}.json"
        json_path = SUBMIT_DIR / json_filename
        
        with open(json_path, "w") as f:
            json.dump(zoom_boxes, f, indent=2)
        
        print(f"[ANALYSIS] Zoom boxes JSON saved: {json_path}")
        print(f"[ANALYSIS] Zoom boxes: {zoom_boxes}")
        
        return json_path


class SimpleMicroscopeController:
    """Simplified microscope controller with integrated analysis"""
    
    def __init__(self, epics_pv=DEFAULT_SCAN_PV, status_pv=DEFAULT_STATUS_PV, fine_scan_params_path="fine_scan_params.json", controls_json_path=None):
        print(f"[MAIN] Initializing simplified microscope controller...")
        
        # Create directories
        TIFF_DIR.mkdir(exist_ok=True)
        SUBMIT_DIR.mkdir(exist_ok=True)
        
        # Use default if not provided
        if controls_json_path is None:
            controls_json_path = CONTROLS_JSON_PATH
        
        # Initialize components
        self.epics_manager = SimpleEPICSManager(epics_pv, status_pv)
        self.queue_manager = SimpleQueueManager(controls_json_path) # Pass controls_json_path
        self.analysis = CustomAnalysis()
        self.fine_scan_params_path = fine_scan_params_path
        self.controls_json_path = controls_json_path
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = False
        self.loop_count = 0
        self.stats = {
            "coarse_scans": 0,
            "zoom_scans": 0,
            "completed_loops": 0
        }
        
        print(f"[MAIN] Simplified microscope controller initialized")
    
    def start(self):
        """Start the autonomous loop"""
        if not self.epics_manager.connected:
            print(f"[MAIN] Cannot start - EPICS not connected")
            return
        
        self.running = True
        self.loop_count = 0
        print(f"[MAIN] Starting autonomous loop...")
        self.run_loop()
    
    def stop(self):
        """Stop the autonomous loop"""
        self.running = False
        print(f"[MAIN] Stopping autonomous loop...")
    
    def run_loop(self):
        """Main loop: coarse scan → wait → analyze → zoom scans → repeat"""
        if not self.running:
            return
        
        self.loop_count += 1
        print(f"[MAIN] Starting loop {self.loop_count}")
        
        # Step 1: Submit coarse scan
        self.submit_coarse_scan()
    
    def submit_coarse_scan(self):
        """Submit coarse scan to queue using controls.json"""
        print(f"[MAIN] Submitting coarse scan...")
        
        # Load scan parameters from controls.json
        if not self.queue_manager.microscope_config:
            print(f"[MAIN] Aborting scan submission due to missing controls.json or fields.")
            time.sleep(5)
            self.run_loop()
            return
        
        # Build scan plan from controls.json
        scan_plan = {
            "label": self.queue_manager.microscope_config.get("label", f"scan_{int(time.time())}"),
            "det_name": self.queue_manager.microscope_config.get("det_name"),
            "mot1": self.queue_manager.microscope_config.get("mot1"),
            "mot1_s": self.queue_manager.microscope_config.get("mot1_s"),
            "mot1_e": self.queue_manager.microscope_config.get("mot1_e"),
            "mot1_n": self.queue_manager.microscope_config.get("mot1_n"),
            "mot2": self.queue_manager.microscope_config.get("mot2"),
            "mot2_s": self.queue_manager.microscope_config.get("mot2_s"),
            "mot2_e": self.queue_manager.microscope_config.get("mot2_e"),
            "mot2_n": self.queue_manager.microscope_config.get("mot2_n"),
            "exp_t": self.queue_manager.microscope_config.get("exp_t"),
            "timestamp": time.time()
        }
        missing = [k for k in ["mot1","mot1_s","mot1_e","mot1_n","mot2","mot2_s","mot2_e","mot2_n","exp_t"] if scan_plan[k] is None]
        if missing:
            print(f"[CONTROLS] ERROR: Missing required fields in controls.json: {missing}")
            time.sleep(5)
            self.run_loop()
            return
        
        if self.queue_manager.submit_coarse_scan(self.controls_json_path):
            self.stats["coarse_scans"] += 1
            print(f"[MAIN] Coarse scan {scan_plan['label']} submitted")
            # Wait for completion
            self.wait_for_coarse_completion(scan_plan["label"])
        else:
            print(f"[MAIN] Failed to submit coarse scan")
            time.sleep(5)
            self.run_loop()
    
    def wait_for_coarse_completion(self, scan_id):
        """Wait for coarse scan to complete"""
        print(f"[MAIN] Waiting for coarse scan {scan_id} to complete...")
        
        # Wait for scan completion using EPICS status
        if self.epics_manager.wait_for_scan_completion():
            print(f"[MAIN] Coarse scan {scan_id} completed")
            self.analyze_coarse_data(scan_id)
        else:
            print(f"[MAIN] Coarse scan {scan_id} failed or timed out")
            # Retry the loop
            time.sleep(5)
            self.run_loop()
    
    def analyze_coarse_data(self, scan_id):
        """Analyze coarse scan data and create zoom boxes"""
        print(f"[MAIN] Analyzing coarse scan data for scan {scan_id}...")
        
        try:
            # Load scan data as numpy array
            data = self.analysis.load_scan_data(scan_id)
            
            # Perform custom analysis
            regions = self.analysis.analyze_data(data, scan_id)
            
            if regions:
                # Create zoom boxes JSON file
                json_path = self.analysis.create_zoom_boxes_json(regions, scan_id)
                
                # Submit zoom scans using the new function
                microscope_config = load_microscope_config(self.controls_json_path)
                self.queue_manager.submit_zoom_scans(str(json_path), microscope_config, self.fine_scan_params_path)
                self.stats["zoom_scans"] += len(regions)
                print(f"[MAIN] {len(regions)} zoom scans submitted")
                
                # Wait for zoom scans to complete
                print(f"[MAIN] Zoom scans have been queued. Waiting for completion should be handled by the main loop if needed.")
                # self.wait_for_zoom_completion(scan_id)  # Removed: scan_id is not defined in this context
            else:
                print(f"[MAIN] No regions found, skipping zoom scans")
                self.complete_loop()
                
        except Exception as e:
            print(f"[MAIN] Error analyzing data: {e}")
            self.complete_loop()
    
    def create_zoom_scans(self, json_file_path_boxes):
        """
        Submit zoom scan plans from a JSON file generated by the analysis class.
        - json_file_path: path to the zoom boxes JSON file
        - microscope_config: dict mapping logical to hardware motor names
        - fine_scan_params_path: path to the fine scan parameter JSON file
        """
        from bluesky_queueserver_api import BPlan, REManagerAPI
        RM = REManagerAPI()

        with open(self.fine_scan_params_path, "r") as f:
            fine_params = json.load(f)

        x_motor = fine_params["mot1"]
        y_motor = fine_params["mot2"]
        exp_t = fine_params["exp_t"]
        det_name = fine_params["det_name"]
        step_size_um = fine_params["step_size_um"]

        with open(json_file_path_boxes, "r") as f:
            boxes = json.load(f)

        for label, info in boxes.items():
            cx, cy = info["real_center_um"]         # center in um
            sx, sy = info["real_size_um"]           # size in um
            num_x = int(sx / step_size_um)
            num_y = int(sy / step_size_um)

            # Define relative scan range around center
            x_start = -sx / 2
            x_end = sx / 2
            y_start = -sy / 2
            y_end = sy / 2

            # Create ROI dictionary to move motors first
            roi = {x_motor: cx, y_motor: cy}

            # Submit to Bluesky queue
            RM.item_add(BPlan(
                "recover_pos_and_scan",
                label,
                roi,
                det_name,
                x_motor,
                x_start,
                x_end,
                num_x,
                y_motor,
                y_start,
                y_end,
                num_y,
                exp_t
            ))
            print(f"[QUEUE] Queued zoom scan: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")

            # Wait for zoom scans to complete
            # self.wait_for_zoom_completion(scan_id) # Removed: scan_id is not defined in this context
        else:
            print(f"[MAIN] Failed to submit zoom scans")
            self.complete_loop()
    
    def wait_for_zoom_completion(self):
        """Wait for all zoom scans to complete using EPICS and queue_get."""
        print(f"[MAIN] Waiting for zoom scans to complete...")
        RM = REManagerAPI()
        while True:
            # Wait for scan completion using EPICS status
            if self.epics_manager.wait_for_scan_completion():
                # Check if any zoom scans remain in the queue
                queue_items = RM.queue_get()["items"]
                zoom_plans = [item for item in queue_items if item.get("item_type") == "plan" and item.get("args", [None])[0] == "recover_pos_and_scan"]
                if zoom_plans:
                    print(f"[MAIN] {len(zoom_plans)} zoom scans still in queue, waiting for next completion...")
                    continue  # Wait for the next scan to complete
                else:
                    print(f"[MAIN] All zoom scans completed")
                    break
            else:
                print(f"[MAIN] Zoom scans failed or timed out")
                break
        self.complete_loop()
    
    def complete_loop(self):
        """Complete the current loop and start the next one"""
        print(f"[MAIN] Loop {self.loop_count} completed")
        self.stats["completed_loops"] += 1
        self.print_stats()
        
        # Start next loop after delay
        time.sleep(2)
        self.run_loop()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[MAIN] Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def print_stats(self):
        """Print current statistics"""
        print(f"\n[STATS] System Statistics:")
        print(f"  Coarse Scans: {self.stats['coarse_scans']}")
        print(f"  Zoom Scans: {self.stats['zoom_scans']}")
        print(f"  Completed Loops: {self.stats['completed_loops']}")
        print(f"  Current Loop: {self.loop_count}")


CONTROLS_JSON_PATH = Path(__file__).parent.parent / "controls.json"

def load_controls_json():
    """Load scan parameters from controls.json in parent directory"""
    try:
        with open(CONTROLS_JSON_PATH, "r") as f:
            params = json.load(f)
        print(f"[CONTROLS] Loaded scan parameters from {CONTROLS_JSON_PATH}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        return params
    except Exception as e:
        print(f"[CONTROLS] ERROR: Could not load controls.json: {e}")
        return None


# Utility function to load microscope_config

def load_microscope_config(controls_path):
    with open(controls_path) as f:
        controls = json.load(f)
    return controls["microscope_config"]

# Update load_coarse_scan_params to accept a path argument

def load_coarse_scan_params(params_path):
    with open(params_path) as f:
        params = json.load(f)
    return params

# Update all calls to submit_coarse_scan to pass the coarse_scan_params_path argument
# In SimpleMicroscopeController, update submit_coarse_scan and any call sites accordingly.


def load_fine_scan_params(params_path):
    with open(params_path) as f:
        params = json.load(f)
    return params


def submit_zoom_scans_from_json(json_file_path, microscope_config, fine_scan_params_path):
    """
    Submit zoom scan plans from a JSON file generated by the analysis class.
    - json_file_path: path to the zoom boxes JSON file
    - microscope_config: dict mapping logical to hardware motor names
    - fine_scan_params_path: path to the fine scan parameter JSON file
    """
    from bluesky_queueserver_api import BPlan, REManagerAPI
    RM = REManagerAPI()

    fine_params = load_fine_scan_params(fine_scan_params_path)
    x_motor = microscope_config[fine_params["mot1"]]
    y_motor = microscope_config[fine_params["mot2"]]
    exp_t = fine_params["exp_t"]
    det_name = fine_params["det_name"]

    with open(json_file_path, "r") as f:
        boxes = json.load(f)

    for label, info in boxes.items():
        cx, cy = info["real_center_um"]         # center in um
        sx, sy = info["real_size_um"]           # size in um
        num_x = int(sx)
        num_y = int(sy)

        # Define relative scan range around center
        x_start = -sx / 2
        x_end = sx / 2
        y_start = -sy / 2
        y_end = sy / 2

        # Create ROI dictionary to move motors first
        roi = {x_motor: cx, y_motor: cy}

        # Submit to Bluesky queue
        RM.item_add(BPlan(
            "recover_pos_and_scan",
            label,
            roi,
            det_name,
            x_motor,
            x_start,
            x_end,
            num_x,
            y_motor,
            y_start,
            y_end,
            num_y,
            exp_t
        ))
        print(f"[QUEUE] Queued zoom scan: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simplified Hardware Queue-based Autonomous Microscope Controller")
    parser.add_argument("--epics-pv", default=DEFAULT_SCAN_PV, help="EPICS scan ID PV name")
    parser.add_argument("--status-pv", default=DEFAULT_STATUS_PV, help="EPICS scan status PV name")
    parser.add_argument("--no-auto-start", action="store_true", help="Don't start automatically")
    
    args = parser.parse_args()
    
    print(f"[MAIN] Simplified hardware queue-based auto-map system starting...")
    print(f"[MAIN] EPICS scan ID PV: {args.epics_pv}")
    print(f"[MAIN] EPICS scan status PV: {args.status_pv}")
    print(f"[MAIN] Auto-start: {not args.no_auto_start}")
    
    # Create controller
    controller = SimpleMicroscopeController(
        epics_pv=args.epics_pv,
        status_pv=args.status_pv
    )
    
    if not args.no_auto_start:
        controller.start()
    
    print(f"[MAIN] System ready. Press Ctrl+C to stop.")
    
    # Keep the application running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[MAIN] Keyboard interrupt received")
        controller.stop()


if __name__ == "__main__":
    main() 


