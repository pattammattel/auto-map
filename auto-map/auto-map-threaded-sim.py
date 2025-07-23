#!/usr/bin/env python3
"""
Threaded and Simulated Autonomous Microscope Controller
- Uses PyQt5 threading for non-blocking operation
- Includes simulation classes for testing without hardware
- Maintains compatibility with existing EPICS and Bluesky integration
"""

import sys
import os
import json
import time
import numpy as np
import argparse
from pathlib import Path
from PyQt5 import QtCore, QtWidgets

# Bluesky imports for simulation
from ophyd.sim import det4, motor1, motor2
from bluesky.plans import grid_scan
from bluesky import RunEngine

# Configuration
TIFF_DIR = Path("./tiff_dir")
SUBMIT_DIR = Path("./submitted")

# EPICS Configuration
DEFAULT_SCAN_PV = "SCAN:ID"
DEFAULT_STATUS_PV = "SCAN:RUNNING"

print(f"[INIT] Threaded and simulated auto-map system starting...")
print(f"[INIT] TIFF directory: {TIFF_DIR}")
print(f"[INIT] Submit directory: {SUBMIT_DIR}")
print(f"[INIT] Default scan ID PV: {DEFAULT_SCAN_PV}")
print(f"[INIT] Default status PV: {DEFAULT_STATUS_PV}")


# --- Define Qt signals ---
class MicroscopeSignals(QtCore.QObject):
    """Qt-based signals for microscope events"""
    coarse_scan_submitted = QtCore.pyqtSignal(str)
    coarse_scan_completed = QtCore.pyqtSignal(str)
    zoom_scan_queued = QtCore.pyqtSignal(object)
    loop_completed = QtCore.pyqtSignal(int)
    error_occurred = QtCore.pyqtSignal(str)


class SimpleEPICSManager:
    """Simple EPICS manager using caget for scan numbers"""
    
    def __init__(self, scan_pv_name=DEFAULT_SCAN_PV, status_pv_name=DEFAULT_STATUS_PV):
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
            import epics
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
            import epics
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


class SimpleQueueManager(QtCore.QObject):
    """Simple queue manager for scan submission, with persistent controls.json config."""
    
    zoom_scan_completed = QtCore.pyqtSignal()
    
    def __init__(self, controls_json_path):
        super().__init__()
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

    def submit_zoom_scans(self, json_file_path_boxes, fine_scan_params_path):
        """
        Submit zoom scan plans from a JSON file generated by the analysis class.
        - json_file_path: path to the zoom boxes JSON file
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
            self.zoom_scan_completed.emit()
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


# --- QThread worker for the loop ---
class LoopWorker(QtCore.QThread):
    """Thread managing the scan loop"""
    def __init__(self, epics_manager, queue_manager, analysis,
                 fine_scan_params_path, controls_json_path, signals):
        super().__init__()
        self.epics_manager = epics_manager
        self.queue_manager = queue_manager
        self.analysis = analysis
        self.fine_scan_params_path = fine_scan_params_path
        self.controls_json_path = controls_json_path
        self.signals = signals
        self._running = False

    def start(self):
        self._running = True
        super().start()

    def run(self):  # this executes in the new thread
        loop_count = 0
        while self._running:
            try:
                loop_count += 1
                scan_id = self.submit_coarse_scan()
                if self.epics_manager.wait_for_scan_completion():
                    self.signals.coarse_scan_completed.emit(scan_id)
                    data = self.analysis.load_scan_data(scan_id)
                    regions = self.analysis.analyze_data(data, scan_id)
                    json_path = self.analysis.create_zoom_boxes_json(regions, scan_id)
                    for region in regions:
                        self.signals.zoom_scan_queued.emit(region)
                    self.queue_manager.submit_zoom_scans(
                        str(json_path),
                        self.fine_scan_params_path
                    )
                self.signals.loop_completed.emit(loop_count)
                self.msleep(1000)  # sleep 1s in ms
            except Exception as e:
                self.signals.error_occurred.emit(str(e))
                self.msleep(5000)  # wait 5s on error

    def stop(self):
        self._running = False
        self.wait()

    def submit_coarse_scan(self):
        params = load_coarse_scan_params(self.controls_json_path)
        label = params.get("label", f"scan_{int(time.time())}")
        self.queue_manager.submit_coarse_scan(self.controls_json_path)
        self.signals.coarse_scan_submitted.emit(label)
        return label


# --- Controller implementation ---
class SimpleMicroscopeController(QtCore.QObject):
    """Microscope controller using PyQt5 signals/slots and threads"""
    def __init__(self, epics_manager, queue_manager, analysis,
                 fine_scan_params_path, controls_json_path, signals=None):
        super().__init__()
        self.signals = signals or MicroscopeSignals()
        self.worker = LoopWorker(
            epics_manager, queue_manager, analysis,
            fine_scan_params_path, controls_json_path, self.signals
        )

    def start(self):
        self.worker.start()

    def stop(self):
        self.worker.stop()


# --- Simulation classes ---
class SimulationEPICSManager:
    """Simulate EPICS PVs and scan completions"""
    def __init__(self, delay=2):
        self.delay = delay
        self.connected = True
        print(f"[SIM] EPICS simulation initialized with {delay}s delay")

    def test_connection(self):
        return True

    def wait_for_scan_completion(self, timeout=10):
        time.sleep(self.delay)
        return True

    def get_current_scan_id(self):
        return int(time.time())

    def get_scan_status(self):
        return "completed"


class SimulationQueueManager(SimpleQueueManager):
    """Simulate queue server connectivity"""
    def __init__(self, controls_json_path):
        super().__init__(controls_json_path)
        self.queue_connected = True
        print(f"[SIM] Queue simulation initialized")

    def submit_coarse_scan(self, coarse_scan_params_path):
        print(f"[SIM] Simulated coarse scan submission")
        return True

    def submit_zoom_scans(self, json_file_path_boxes, fine_scan_params_path):
        print(f"[SIM] Simulated zoom scan submission")
        self.zoom_scan_completed.emit()
        return True


class SimulationScanManager(QtCore.QObject):
    """Simulate scan by running a real grid_scan with simulated hardware in main thread."""
    scan_requested = QtCore.pyqtSignal(str)  # Signal to request scan from main thread
    
    def __init__(self):
        super().__init__()
        self.RE = RunEngine({})
        self.scan_completed = False
        self.scan_result = None

    def submit_coarse_scan(self, *args, **kwargs):
        print("[SIM] Requesting coarse grid scan from main thread")
        self.scan_completed = False
        self.scan_requested.emit("coarse")
        # Wait for scan to complete
        while not self.scan_completed:
            time.sleep(0.1)
        return self.scan_result

    def submit_zoom_scans(self, *args, **kwargs):
        print("[SIM] Requesting zoom grid scan from main thread")
        self.scan_completed = False
        self.scan_requested.emit("zoom")
        # Wait for scan to complete
        while not self.scan_completed:
            time.sleep(0.1)
        return self.scan_result

    @QtCore.pyqtSlot(str)
    def run_scan(self, scan_type):
        try:
            if scan_type == "coarse":
                print("[SIM] Running simulated coarse grid scan with ophyd.sim devices")
                self.RE(grid_scan([det4],
                                  motor1, -1.5, 1.5, 30,
                                  motor2, -0.1, 0.1, 50))
            elif scan_type == "zoom":
                print("[SIM] Running simulated zoom grid scan with ophyd.sim devices")
                self.RE(grid_scan([det4],
                                  motor1, -0.5, 0.5, 2,
                                  motor2, -0.05, 0.05, 2))
            self.scan_result = True
        except Exception as e:
            print(f"[SIM] Error running scan: {e}")
            self.scan_result = False
        finally:
            self.scan_completed = True


# Utility functions
def load_controls_json():
    """Load scan parameters from controls.json in parent directory"""
    try:
        controls_path = Path(__file__).parent.parent / "controls.json"
        with open(controls_path, "r") as f:
            params = json.load(f)
        print(f"[CONTROLS] Loaded scan parameters from {controls_path}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        return params
    except Exception as e:
        print(f"[CONTROLS] ERROR: Could not load controls.json: {e}")
        return None


def load_microscope_config(controls_path):
    with open(controls_path) as f:
        controls = json.load(f)
    return controls["microscope_config"]


def load_coarse_scan_params(params_path):
    with open(params_path) as f:
        params = json.load(f)
    return params


def load_fine_scan_params(params_path):
    with open(params_path) as f:
        params = json.load(f)
    return params


def main():
    """Main function with simulation and threading support"""
    parser = argparse.ArgumentParser(description="Threaded and Simulated Autonomous Microscope Controller")
    parser.add_argument("--epics-pv", default=DEFAULT_SCAN_PV, help="EPICS scan ID PV name")
    parser.add_argument("--status-pv", default=DEFAULT_STATUS_PV, help="EPICS scan status PV name")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--sim-delay", type=float, default=1.0, help="Simulation delay in seconds")
    parser.add_argument("--no-auto-start", action="store_true", help="Don't start automatically")
    
    args = parser.parse_args()
    
    print(f"[MAIN] Threaded and simulated auto-map system starting...")
    print(f"[MAIN] Simulation mode: {args.simulate}")
    print(f"[MAIN] Simulation delay: {args.sim_delay}s")
    print(f"[MAIN] Auto-start: {not args.no_auto_start}")
    
    # Create directories
    TIFF_DIR.mkdir(exist_ok=True)
    SUBMIT_DIR.mkdir(exist_ok=True)
    
    # Initialize components based on mode
    if args.simulate:
        epics_manager = SimulationEPICSManager(delay=args.sim_delay)
        queue_manager = SimulationScanManager()  # <--- use this!
        # Connect scan request signal to run_scan slot
        queue_manager.scan_requested.connect(queue_manager.run_scan)
    else:
        epics_manager = SimpleEPICSManager(args.epics_pv, args.status_pv)
        queue_manager = SimpleQueueManager("controls.json")
    
    analysis = CustomAnalysis()
    
    # Create signals and connect slots
    signals = MicroscopeSignals()
    signals.coarse_scan_submitted.connect(lambda sid: print(f"[SIGNAL] Coarse scan submitted: {sid}"))
    signals.coarse_scan_completed.connect(lambda sid: print(f"[SIGNAL] Coarse scan completed: {sid}"))
    signals.zoom_scan_queued.connect(lambda reg: print(f"[SIGNAL] Zoom scan queued for region: {reg}"))
    signals.loop_completed.connect(lambda cnt: print(f"[SIGNAL] Loop {cnt} completed"))
    signals.error_occurred.connect(lambda err: print(f"[SIGNAL] Error: {err}"))
    
    # Create controller
    controller = SimpleMicroscopeController(
        epics_manager=epics_manager,
        queue_manager=queue_manager,
        analysis=analysis,
        fine_scan_params_path="fine_scan_params.json",
        controls_json_path="controls.json",
        signals=signals
    )
    
    if not args.no_auto_start:
        controller.start()
    
    print(f"[MAIN] System ready. Press Ctrl+C to stop.")
    
    # Create QApplication instance (required for QThread and signals/slots)
    app = QtWidgets.QApplication(sys.argv)
    
    # Keep the application running
    try:
        app.exec_()
    except KeyboardInterrupt:
        print(f"\n[MAIN] Keyboard interrupt received")
        controller.stop()


if __name__ == "__main__":
    main() 