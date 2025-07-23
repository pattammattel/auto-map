import sys
import os
import json
import zmq
import threading
import time
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QListWidget, QTextEdit, QHBoxLayout, QLineEdit
)
from PyQt5.QtCore import (
    QObject, pyqtSignal, QTimer, QFileSystemWatcher, QThread
)

# Configuration
TIFF_DIR = Path("./tiff_dir")
SUBMIT_DIR = Path("./submitted")
ZOOM_STEP_SIZE = 0.1

# ZMQ Configuration
SCAN_PORT = 5555
ANALYSIS_PORT = 5556
CONTROL_PORT = 5557

# EPICS Configuration
DEFAULT_SCAN_PV = "SCAN:ID"  # Default scan ID PV name

print(f"[INIT] Auto-map ZMQ system starting...")
print(f"[INIT] TIFF directory: {TIFF_DIR}")
print(f"[INIT] Submit directory: {SUBMIT_DIR}")
print(f"[INIT] ZMQ ports - Scan: {SCAN_PORT}, Analysis: {ANALYSIS_PORT}, Control: {CONTROL_PORT}")
print(f"[INIT] EPICS scan PV: {DEFAULT_SCAN_PV}")


class EPICSManager(QObject):
    """Manages EPICS integration for scan ID synchronization"""
    
    scan_id_updated = pyqtSignal(int)
    epics_connected = pyqtSignal(bool)
    
    def __init__(self, scan_pv_name=DEFAULT_SCAN_PV):
        super().__init__()
        self.scan_pv_name = scan_pv_name
        self.epics_connected_status = False
        self.current_scan_id = 0
        self.pv_monitor_timer = QTimer()
        self.pv_monitor_timer.timeout.connect(self.check_scan_id)
        
        # Try to import EPICS components
        try:
            global caget
            from epics import caget
            self.epics_available = True
            print(f"[EPICS] EPICS integration initialized with PV: {scan_pv_name}")
        except ImportError as e:
            print(f"[EPICS] Warning: Could not import EPICS components: {e}")
            print(f"[EPICS] Scan ID will be simulated")
            self.epics_available = False
    
    def start_monitoring(self):
        """Start monitoring the scan ID PV"""
        if self.epics_available:
            self.pv_monitor_timer.start(1000)  # Check every second
            print(f"[EPICS] Started monitoring scan ID PV: {self.scan_pv_name}")
        else:
            print(f"[EPICS] Running in simulation mode")
    
    def stop_monitoring(self):
        """Stop monitoring the scan ID PV"""
        self.pv_monitor_timer.stop()
        print(f"[EPICS] Stopped monitoring scan ID PV")
    
    def check_scan_id(self):
        """Check current scan ID from EPICS PV"""
        try:
            if not self.epics_available:
                # Simulation mode - increment scan ID
                self.current_scan_id += 1
                self.scan_id_updated.emit(self.current_scan_id)
                return
            
            # Get scan ID from EPICS PV
            scan_id = caget(self.scan_pv_name)
            
            if scan_id is not None and scan_id != self.current_scan_id:
                self.current_scan_id = int(scan_id)
                print(f"[EPICS] Scan ID updated: {self.current_scan_id}")
                self.scan_id_updated.emit(self.current_scan_id)
                self.epics_connected.emit(True)
            elif scan_id is None:
                print(f"[EPICS] Warning: Could not read PV {self.scan_pv_name}")
                self.epics_connected.emit(False)
                
        except Exception as e:
            print(f"[EPICS] Error reading scan ID: {e}")
            self.epics_connected.emit(False)
    
    def get_current_scan_id(self):
        """Get the current scan ID"""
        return self.current_scan_id
    
    def set_scan_pv(self, pv_name):
        """Set a new scan PV name"""
        self.scan_pv_name = pv_name
        print(f"[EPICS] Scan PV changed to: {pv_name}")
    
    def test_connection(self):
        """Test EPICS connection"""
        if not self.epics_available:
            return {"status": "simulated", "scan_id": self.current_scan_id}
        
        try:
            scan_id = caget(self.scan_pv_name)
            if scan_id is not None:
                return {"status": "connected", "scan_id": int(scan_id)}
            else:
                return {"status": "error", "error": f"Could not read PV {self.scan_pv_name}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class BlueskyQueueManager(QObject):
    """Manages Bluesky queue server integration"""
    
    scan_submitted = pyqtSignal(int, str)
    scan_failed = pyqtSignal(int, str)
    
    def __init__(self):
        super().__init__()
        self.queue_connected = False
        self.submitted_scans = []
        
        # Try to import Bluesky components
        try:
            global RM, BPlan
            from bluesky_queueserver_api import BPlan
            from bluesky_queueserver_api.zmq import REManagerAPI as RM
            self.queue_connected = True
            print(f"[BLUESKY] Queue server integration initialized")
        except ImportError as e:
            print(f"[BLUESKY] Warning: Could not import Bluesky components: {e}")
            print(f"[BLUESKY] Queue submission will be simulated")
            self.queue_connected = False
    
    def send_fly2d_to_queue(self, label, dets, mot1, mot1_s, mot1_e, mot1_n, mot2, mot2_s, mot2_e, mot2_n, exp_t):
        """Submit a fly2d scan to the Bluesky queue server"""
        try:
            if not self.queue_connected:
                print(f"[BLUESKY] Simulating queue submission for scan: {label}")
                self.submitted_scans.append({
                    "label": label,
                    "type": "fly2d",
                    "motors": [mot1, mot2],
                    "ranges": [(mot1_s, mot1_e), (mot2_s, mot2_e)],
                    "steps": [mot1_n, mot2_n],
                    "exposure": exp_t,
                    "timestamp": time.time()
                })
                self.scan_submitted.emit(len(self.submitted_scans), label)
                return True
            
            det_names = [d.name for d in eval(dets)]
            RM.item_add((BPlan("fly2d_qserver_plan",
                    label,
                    det_names, 
                    mot1, 
                    mot1_s, 
                    mot1_e, 
                    mot1_n, 
                    mot2, 
                    mot2_s, 
                    mot2_e, 
                    mot2_n, 
                    exp_t)))
            
            print(f"[BLUESKY] Successfully submitted fly2d scan: {label}")
            self.scan_submitted.emit(len(self.submitted_scans), label)
            return True
            
        except Exception as e:
            error_msg = f"Failed to submit scan {label}: {str(e)}"
            print(f"[BLUESKY] {error_msg}")
            self.scan_failed.emit(0, error_msg)
            return False
    
    def send_json_boxes_to_queue(self, json_file_path, dets="dets1", x_motor="zpssx", y_motor="zpssy", exp_t=0.01, px_per_um=1.25):
        """
        For each region in the JSON file:
        - Move stage to real_center_um
        - Perform fly2d scan centered on that position
        """
        try:
            if not self.queue_connected:
                print(f"[BLUESKY] Simulating JSON queue submission from: {json_file_path}")
                # Simulate the submission
                with open(json_file_path, "r") as f:
                    boxes = json.load(f)
                
                for label, info in boxes.items():
                    self.submitted_scans.append({
                        "label": label,
                        "type": "fly2d_json",
                        "center": info["real_center_um"],
                        "size": info["real_size_um"],
                        "timestamp": time.time()
                    })
                    print(f"[BLUESKY] Simulated: {label} | center {info['real_center_um']} µm | size {info['real_size_um']} µm")
                
                return True
            
            with open(json_file_path, "r") as f:
                boxes = json.load(f)
     
            for label, info in boxes.items():
                cx, cy = info["real_center_um"]         # center in um
                sx, sy = info["real_size_um"]           # size in um
                num_x = int(sx * px_per_um)
                num_y = int(sy * px_per_um)
     
                # Define relative scan range around center
                x_start = -sx / 2
                x_end = sx / 2
                y_start = -sy / 2
                y_end = sy / 2
     
                # Detector names
                det_names = [d.name for d in eval(dets)]
     
                # Create ROI dictionary to move motors first
                roi = {x_motor: cx, y_motor: cy}
     
                RM.item_add(BPlan(
                    "recover_pos_and_scan",
                    label,
                    roi,
                    det_names,
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
                print(f"[BLUESKY] Queued: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to submit JSON boxes from {json_file_path}: {str(e)}"
            print(f"[BLUESKY] {error_msg}")
            return False
    
    def submit_zoom_scan(self, scan_plan):
        """Submit a zoom scan plan to the queue (legacy method)"""
        try:
            # Extract parameters from scan plan
            scan_id = scan_plan["id"]
            label = f"zoom_scan_{scan_id}"
            
            # Default parameters - these should be configured based on your setup
            dets = "['det1']"  # Replace with actual detector list
            mot1 = "x"  # X motor name
            mot2 = "y"  # Y motor name
            mot1_s = scan_plan["x_start"]
            mot1_e = scan_plan["x_end"]
            mot1_n = int((mot1_e - mot1_s) / scan_plan["step_size"]) + 1
            mot2_s = scan_plan["y_start"]
            mot2_e = scan_plan["y_end"]
            mot2_n = int((mot2_e - mot2_s) / scan_plan["step_size"]) + 1
            exp_t = 0.1  # Exposure time in seconds
            
            print(f"[BLUESKY] Submitting zoom scan {scan_id}:")
            print(f"[BLUESKY]   X: {mot1_s:.3f} to {mot1_e:.3f} ({mot1_n} steps)")
            print(f"[BLUESKY]   Y: {mot2_s:.3f} to {mot2_e:.3f} ({mot2_n} steps)")
            print(f"[BLUESKY]   Step size: {scan_plan['step_size']}")
            print(f"[BLUESKY]   Exposure: {exp_t}s")
            
            return self.send_fly2d_to_queue(
                label, dets, mot1, mot1_s, mot1_e, mot1_n, 
                mot2, mot2_s, mot2_e, mot2_n, exp_t
            )
            
        except Exception as e:
            error_msg = f"Failed to prepare zoom scan {scan_plan.get('id', 'unknown')}: {str(e)}"
            print(f"[BLUESKY] {error_msg}")
            self.scan_failed.emit(scan_plan.get('id', 0), error_msg)
            return False
    
    def get_queue_status(self):
        """Get current queue status"""
        if not self.queue_connected:
            return {"status": "simulated", "submitted": len(self.submitted_scans)}
        
        try:
            # This would return actual queue status
            return {"status": "connected", "submitted": len(self.submitted_scans)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class ZMQMessageHandler(QObject):
    """Handles ZMQ communication between components"""
    
    scan_request_received = pyqtSignal(dict)
    analysis_result_received = pyqtSignal(dict)
    control_message_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.context = zmq.Context()
        self.running = False
        
        # Control socket (PUB/SUB for system-wide messages)
        self.control_pub = self.context.socket(zmq.PUB)
        self.control_pub.bind(f"tcp://*:{CONTROL_PORT}")
        
        # Analysis socket (REQ/REP for analysis requests)
        self.analysis_socket = self.context.socket(zmq.REQ)
        self.analysis_socket.connect(f"tcp://localhost:{ANALYSIS_PORT}")
        
        print(f"[ZMQ] Message handler initialized")
    
    def start(self):
        self.running = True
        print(f"[ZMQ] Message handler started")
    
    def stop(self):
        self.running = False
        self.control_pub.close()
        self.analysis_socket.close()
        print(f"[ZMQ] Message handler stopped")
    
    def send_control_message(self, message_type, data=None):
        """Send control message to all subscribers"""
        message = {
            "type": message_type,
            "timestamp": time.time(),
            "data": data or {}
        }
        self.control_pub.send_json(message)
        print(f"[ZMQ] Control message sent: {message_type}")
    
    def send_tiff_data_for_analysis(self, tiff_array, metadata):
        """Send TIFF array and metadata to analysis service"""
        try:
            # Prepare the analysis request
            request = {
                "type": "analyze_tiff_array",
                "tiff_array": tiff_array.tolist(),  # Convert numpy array to list for JSON
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            print(f"[ZMQ] Sending TIFF array for analysis (shape: {tiff_array.shape})")
            print(f"[ZMQ] Metadata: {metadata}")
            
            # Send the request
            self.analysis_socket.send_json(request)
            
            # Wait for response
            response = self.analysis_socket.recv_json()
            print(f"[ZMQ] Analysis response received: {response}")
            return response
            
        except Exception as e:
            print(f"[ZMQ] Error sending TIFF data for analysis: {e}")
            return None
    
    def send_json_path_for_queue(self, json_path, metadata=None):
        """Send JSON file path to queue submission service"""
        try:
            # Prepare the queue submission request
            request = {
                "type": "submit_json_boxes",
                "json_path": str(json_path),
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            
            print(f"[ZMQ] Sending JSON path for queue submission: {json_path}")
            
            # Send the request
            self.analysis_socket.send_json(request)
            
            # Wait for response
            response = self.analysis_socket.recv_json()
            print(f"[ZMQ] Queue submission response received: {response}")
            return response
            
        except Exception as e:
            print(f"[ZMQ] Error sending JSON path for queue submission: {e}")
            return None
    
    def request_analysis(self, tiff_path):
        """Request analysis of a TIFF file (legacy method)"""
        request = {
            "type": "analyze_tiff",
            "tiff_path": str(tiff_path),
            "timestamp": time.time()
        }
        print(f"[ZMQ] Sending analysis request for: {tiff_path}")
        self.analysis_socket.send_json(request)
        
        try:
            response = self.analysis_socket.recv_json()
            print(f"[ZMQ] Analysis response received: {response}")
            return response
        except zmq.error.Again:
            print(f"[ZMQ] Analysis request timeout")
            return None


class AnalysisService(QThread):
    """Separate analysis service that processes TIFF arrays and metadata"""
    
    def __init__(self):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{ANALYSIS_PORT}")
        self.running = False
        print(f"[ANALYSIS] Analysis service initialized")
    
    def run(self):
        self.running = True
        print(f"[ANALYSIS] Analysis service started, listening on port {ANALYSIS_PORT}")
        
        while self.running:
            try:
                # Wait for analysis requests
                request = self.socket.recv_json(flags=zmq.NOBLOCK)
                print(f"[ANALYSIS] Received request: {request['type']}")
                
                if request["type"] == "analyze_tiff_array":
                    result = self.analyze_tiff_array(request["tiff_array"], request["metadata"])
                    response = {
                        "type": "analysis_result",
                        "regions": result,
                        "metadata": request["metadata"],
                        "timestamp": time.time()
                    }
                    self.socket.send_json(response)
                    print(f"[ANALYSIS] Array analysis completed, sent response")
                
                elif request["type"] == "analyze_tiff":
                    result = self.analyze_tiff(Path(request["tiff_path"]))
                    response = {
                        "type": "analysis_result",
                        "regions": result,
                        "tiff_path": request["tiff_path"],
                        "timestamp": time.time()
                    }
                    self.socket.send_json(response)
                    print(f"[ANALYSIS] File analysis completed, sent response")
                
                elif request["type"] == "submit_json_boxes":
                    # Handle JSON path submission for queue
                    json_path = Path(request["json_path"])
                    metadata = request.get("metadata", {})
                    
                    if json_path.exists():
                        # Submit to queue using the JSON file
                        success = self.submit_json_boxes_to_queue(json_path, metadata)
                        response = {
                            "type": "queue_submission_result",
                            "success": success,
                            "json_path": str(json_path),
                            "timestamp": time.time()
                        }
                    else:
                        response = {
                            "type": "queue_submission_result",
                            "success": False,
                            "error": f"JSON file not found: {json_path}",
                            "timestamp": time.time()
                        }
                    
                    self.socket.send_json(response)
                    print(f"[ANALYSIS] JSON queue submission completed, sent response")
                
            except zmq.error.Again:
                # No message available, continue
                time.sleep(0.1)
            except Exception as e:
                print(f"[ANALYSIS] Error processing request: {e}")
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        self.socket.close()
        print(f"[ANALYSIS] Analysis service stopped")
    
    def analyze_tiff_array(self, tiff_array, metadata):
        """Analyze TIFF array and return regions of interest"""
        print(f"[ANALYSIS] Analyzing TIFF array with shape: {tiff_array.shape}")
        print(f"[ANALYSIS] Metadata: {metadata}")
        
        # Convert list back to numpy array
        tiff_array = np.array(tiff_array)
        
        # Extract scan positions from metadata
        scan_positions = metadata.get("scan_positions", {})
        x_positions = scan_positions.get("x", [])
        y_positions = scan_positions.get("y", [])
        
        print(f"[ANALYSIS] Scan positions - X: {len(x_positions)} points, Y: {len(y_positions)} points")
        
        # Simulated analysis - in real implementation this would process the TIFF array
        # This could include image processing, feature detection, etc.
        time.sleep(1)  # Simulate processing time
        
        # Generate regions based on array analysis and scan positions
        regions = []
        
        # Example: Find regions with high intensity
        if tiff_array.size > 0:
            # Simple threshold-based region detection
            threshold = np.mean(tiff_array) + 2 * np.std(tiff_array)
            high_intensity_coords = np.where(tiff_array > threshold)
            
            if len(high_intensity_coords[0]) > 0:
                # Group nearby high-intensity points into regions
                for i in range(min(3, len(high_intensity_coords[0]))):  # Max 3 regions
                    y_idx = high_intensity_coords[0][i] if i < len(high_intensity_coords[0]) else 0
                    x_idx = high_intensity_coords[1][i] if i < len(high_intensity_coords[1]) else 0
                    
                    # Convert array indices to scan coordinates
                    if len(x_positions) > x_idx and len(y_positions) > y_idx:
                        x_coord = x_positions[x_idx]
                        y_coord = y_positions[y_idx]
                        
                        regions.append({
                            "x": x_coord,
                            "y": y_coord,
                            "w": 1.0,
                            "h": 1.0,
                            "confidence": 0.85 + i * 0.05,
                            "array_x": x_idx,
                            "array_y": y_idx,
                            "intensity": float(tiff_array[y_idx, x_idx])
                        })
        
        # If no regions found, create some default ones
        if not regions:
            regions = [
                {"x": 10.0, "y": 5.0, "w": 1.0, "h": 1.0, "confidence": 0.85},
                {"x": 15.0, "y": 8.0, "w": 0.5, "h": 0.5, "confidence": 0.92},
                {"x": 7.0, "y": 12.0, "w": 0.8, "h": 0.8, "confidence": 0.78}
            ]
        
        print(f"[ANALYSIS] Analysis complete. Found {len(regions)} regions: {regions}")
        return regions
    
    def analyze_tiff(self, tiff_path):
        """Analyze TIFF file and return regions of interest (legacy method)"""
        print(f"[ANALYSIS] Analyzing {tiff_path}...")
        
        # Simulated analysis - in real implementation this would process the TIFF
        # This could include image processing, feature detection, etc.
        time.sleep(1)  # Simulate processing time
        
        regions = [
            {"x": 10.0, "y": 5.0, "w": 1.0, "h": 1.0, "confidence": 0.85},
            {"x": 15.0, "y": 8.0, "w": 0.5, "h": 0.5, "confidence": 0.92},
            {"x": 7.0, "y": 12.0, "w": 0.8, "h": 0.8, "confidence": 0.78}
        ]
        
        print(f"[ANALYSIS] Analysis complete. Found {len(regions)} regions: {regions}")
        return regions
    
    def submit_json_boxes_to_queue(self, json_path, metadata):
        """Submit JSON boxes to queue using the queue manager"""
        try:
            print(f"[ANALYSIS] Submitting JSON boxes to queue: {json_path}")
            
            # This would typically use a queue manager instance
            # For now, we'll simulate the submission
            success = True  # Placeholder for actual queue submission
            
            print(f"[ANALYSIS] JSON boxes submitted successfully: {success}")
            return success
            
        except Exception as e:
            print(f"[ANALYSIS] Error submitting JSON boxes to queue: {e}")
            return False


class ScanController(QObject):
    """Controls the scanning process and communicates via ZMQ"""
    
    scan_started = pyqtSignal(int)
    scan_completed = pyqtSignal(int)
    zoom_scans_created = pyqtSignal(list)
    zoom_scans_submitted = pyqtSignal(list)
    
    def __init__(self, message_handler, queue_manager, epics_manager):
        super().__init__()
        self.message_handler = message_handler
        self.queue_manager = queue_manager
        self.epics_manager = epics_manager
        self.scan_id = 0
        self.zoom_phase = False
        
        # Connect queue manager signals
        self.queue_manager.scan_submitted.connect(self.on_scan_submitted)
        self.queue_manager.scan_failed.connect(self.on_scan_failed)
        
        # Connect EPICS manager signals
        self.epics_manager.scan_id_updated.connect(self.on_epics_scan_id_updated)
        
        print(f"[SCAN] Scan controller initialized")
    
    def start_coarse_scan(self):
        """Start a new coarse scan"""
        print(f"[SCAN] Starting coarse scan...")
        self.zoom_phase = False
        
        # Get scan ID from EPICS
        epics_scan_id = self.epics_manager.get_current_scan_id()
        self.scan_id = epics_scan_id
        
        scan_plan = {
            "type": "coarse",
            "id": self.scan_id,
            "epics_scan_id": epics_scan_id,
            "x_range": [0, 10],
            "y_range": [0, 10],
            "step_size": 0.5,
            "timestamp": time.time()
        }
        
        self.save_scan_plan(scan_plan)
        self.message_handler.send_control_message("coarse_scan_started", scan_plan)
        self.scan_started.emit(self.scan_id)
        
        print(f"[SCAN] Coarse scan {self.scan_id} started (EPICS ID: {epics_scan_id})")
    
    def on_epics_scan_id_updated(self, new_scan_id):
        """Handle EPICS scan ID updates"""
        print(f"[SCAN] EPICS scan ID updated: {new_scan_id}")
        if not self.zoom_phase:
            # If not in zoom phase, this might be a new coarse scan
            self.scan_id = new_scan_id
    
    def process_tiff_array(self, tiff_array, metadata):
        """Process TIFF array and metadata to generate zoom scans"""
        print(f"[SCAN] Processing TIFF array for zoom scans (shape: {tiff_array.shape})")
        self.zoom_phase = True
        
        # Send TIFF array and metadata for analysis via ZMQ
        analysis_result = self.message_handler.send_tiff_data_for_analysis(tiff_array, metadata)
        
        if analysis_result and analysis_result.get("regions"):
            regions = analysis_result["regions"]
            print(f"[SCAN] Analysis returned {len(regions)} regions")
            self.create_zoom_scans(regions)
        else:
            print(f"[SCAN] No regions found or analysis failed")
            self.zoom_phase = False
    
    def create_zoom_scans(self, regions):
        """Create zoom scan plans based on analyzed regions"""
        print(f"[SCAN] Creating zoom scans for {len(regions)} regions...")
        
        zoom_plans = []
        for i, region in enumerate(regions):
            # Generate a temporary ID for tracking (will be replaced when executed)
            temp_scan_id = f"temp_{int(time.time() * 1000)}_{i}"
            
            zoom_plan = {
                "type": "zoom",
                "id": temp_scan_id,
                "region_index": i,
                "x_start": region["x"] - region["w"] / 2,
                "x_end": region["x"] + region["w"] / 2,
                "y_start": region["y"] - region["h"] / 2,
                "y_end": region["y"] + region["h"] / 2,
                "step_size": ZOOM_STEP_SIZE,
                "confidence": region.get("confidence", 0.0),
                "timestamp": time.time(),
                "status": "submitted"
            }
            
            # Add array position info if available
            if "array_x" in region and "array_y" in region:
                zoom_plan["array_position"] = {
                    "x": region["array_x"],
                    "y": region["array_y"],
                    "intensity": region.get("intensity", 0.0)
                }
            
            self.save_scan_plan(zoom_plan)
            zoom_plans.append(zoom_plan)
            print(f"[SCAN] Created zoom scan {temp_scan_id} for region {i+1}")
        
        self.zoom_scans_created.emit(zoom_plans)
        self.message_handler.send_control_message("zoom_scans_created", {"count": len(zoom_plans)})
        
        # Submit zoom scans to queue
        self.submit_zoom_scans_to_queue(zoom_plans)
    
    def submit_zoom_scans_to_queue(self, zoom_plans):
        """Submit zoom scans to Bluesky queue using JSON files via ZMQ"""
        print(f"[SCAN] Submitting {len(zoom_plans)} zoom scans to queue via ZMQ...")
        
        # Find the most recent zoom boxes JSON file
        json_files = list(SUBMIT_DIR.glob("zoom_boxes_*.json"))
        if not json_files:
            print(f"[SCAN] No zoom boxes JSON files found")
            return
        
        # Get the most recent JSON file
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"[SCAN] Using JSON file: {latest_json}")
        
        # Submit using ZMQ message passing
        metadata = {
            "scan_id": self.scan_id,
            "zoom_plans_count": len(zoom_plans),
            "timestamp": time.time()
        }
        
        response = self.message_handler.send_json_path_for_queue(latest_json, metadata)
        
        if response and response.get("success"):
            print(f"[SCAN] Successfully submitted zoom scans via ZMQ")
            self.zoom_scans_submitted.emit(zoom_plans)
            self.message_handler.send_control_message("zoom_scans_submitted", {"count": len(zoom_plans)})
        else:
            error_msg = response.get("error", "Unknown error") if response else "No response"
            print(f"[SCAN] Failed to submit zoom scans via ZMQ: {error_msg}")
        
        # Schedule completion of zoom phase
        QTimer.singleShot(3000, self.complete_zoom_phase)
    
    def on_scan_submitted(self, scan_count, label):
        """Handle successful scan submission"""
        print(f"[SCAN] Scan submitted to queue: {label} (total: {scan_count})")
    
    def on_scan_failed(self, scan_id, error):
        """Handle failed scan submission"""
        print(f"[SCAN] Scan submission failed: {error}")
    
    def complete_zoom_phase(self):
        """Complete zoom phase and restart coarse scan"""
        print(f"[SCAN] Completing zoom phase...")
        self.zoom_phase = False
        self.message_handler.send_control_message("zoom_phase_completed")
        self.scan_completed.emit(self.scan_id)
        
        # Start next coarse scan
        QTimer.singleShot(1000, self.start_coarse_scan)
    
    def save_scan_plan(self, plan):
        """Save scan plan to JSON file"""
        print(f"[SCAN] Saving scan plan {plan['id']}...")
        SUBMIT_DIR.mkdir(exist_ok=True)
        out_file = SUBMIT_DIR / f"scan_{plan['id']}.json"
        
        with open(out_file, "w") as f:
            json.dump(plan, f, indent=2)
        
        print(f"[SCAN] Plan saved to: {out_file}")


class MicroscopeGUI(QWidget):
    """GUI for the autonomous microscope system"""
    
    def __init__(self):
        super().__init__()
        print(f"[GUI] Initializing GUI...")
        
        self.setWindowTitle("Autonomous Microscope Controller (ZMQ + Bluesky + EPICS)")
        self.resize(800, 700)
        
        # Initialize components
        self.message_handler = ZMQMessageHandler()
        self.analysis_service = AnalysisService()
        self.queue_manager = BlueskyQueueManager()
        self.epics_manager = EPICSManager()
        self.scan_controller = ScanController(self.message_handler, self.queue_manager, self.epics_manager)
        
        self.setup_ui()
        self.setup_connections()
        
        print(f"[GUI] GUI initialization complete")
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Status section
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Initializing...")
        self.scan_id_label = QLabel("Current Scan: None")
        self.queue_status_label = QLabel("Queue: Disconnected")
        self.epics_status_label = QLabel("EPICS: Disconnected")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.scan_id_label)
        status_layout.addWidget(self.queue_status_label)
        status_layout.addWidget(self.epics_status_label)
        layout.addLayout(status_layout)
        
        # EPICS Configuration
        epics_layout = QHBoxLayout()
        epics_layout.addWidget(QLabel("Scan PV:"))
        self.scan_pv_input = QLineEdit(DEFAULT_SCAN_PV)
        self.scan_pv_input.setPlaceholderText("Enter EPICS PV name")
        epics_layout.addWidget(self.scan_pv_input)
        self.epics_test_button = QPushButton("Test EPICS")
        epics_layout.addWidget(self.epics_test_button)
        layout.addLayout(epics_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start System")
        self.stop_button = QPushButton("Stop System")
        self.queue_status_button = QPushButton("Check Queue Status")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.queue_status_button)
        layout.addLayout(button_layout)
        
        # Scan information
        scan_info_layout = QHBoxLayout()
        self.coarse_label = QLabel("Coarse Scans: 0")
        self.zoom_label = QLabel("Zoom Scans: 0")
        self.submitted_label = QLabel("Submitted: 0")
        scan_info_layout.addWidget(self.coarse_label)
        scan_info_layout.addWidget(self.zoom_label)
        scan_info_layout.addWidget(self.submitted_label)
        layout.addLayout(scan_info_layout)
        
        # Zoom scan list
        layout.addWidget(QLabel("Recent Zoom Scans:"))
        self.zoom_list = QListWidget()
        layout.addWidget(self.zoom_list)
        
        # Log display
        layout.addWidget(QLabel("System Log:"))
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        layout.addWidget(self.log_display)
        
        self.setLayout(layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Button connections
        self.start_button.clicked.connect(self.start_system)
        self.stop_button.clicked.connect(self.stop_system)
        self.queue_status_button.clicked.connect(self.check_queue_status)
        self.epics_test_button.clicked.connect(self.test_epics_connection)
        
        # Scan controller connections
        self.scan_controller.scan_started.connect(self.on_scan_started)
        self.scan_controller.scan_completed.connect(self.on_scan_completed)
        self.scan_controller.zoom_scans_created.connect(self.on_zoom_scans_created)
        self.scan_controller.zoom_scans_submitted.connect(self.on_zoom_scans_submitted)
        
        # Message handler connections
        self.message_handler.control_message_received.connect(self.on_control_message)
        
        # EPICS manager connections
        self.epics_manager.epics_connected.connect(self.on_epics_connection_changed)
    
    def start_system(self):
        """Start the autonomous microscope system"""
        print(f"[GUI] Starting system...")
        self.log_message("Starting autonomous microscope system...")
        
        # Update EPICS PV if changed
        new_pv = self.scan_pv_input.text().strip()
        if new_pv:
            self.epics_manager.set_scan_pv(new_pv)
        
        # Start EPICS monitoring
        self.epics_manager.start_monitoring()
        
        # Start analysis service
        self.analysis_service.start()
        
        # Start message handler
        self.message_handler.start()
        
        # Start first coarse scan
        self.scan_controller.start_coarse_scan()
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Running")
        
        # Update status displays
        queue_status = self.queue_manager.get_queue_status()
        self.queue_status_label.setText(f"Queue: {queue_status['status']}")
        
        epics_status = self.epics_manager.test_connection()
        self.epics_status_label.setText(f"EPICS: {epics_status['status']}")
        
        self.log_message("System started successfully")
    
    def stop_system(self):
        """Stop the autonomous microscope system"""
        print(f"[GUI] Stopping system...")
        self.log_message("Stopping autonomous microscope system...")
        
        # Stop components
        self.epics_manager.stop_monitoring()
        self.analysis_service.stop()
        self.message_handler.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        
        self.log_message("System stopped")
    
    def test_epics_connection(self):
        """Test EPICS connection"""
        status = self.epics_manager.test_connection()
        self.epics_status_label.setText(f"EPICS: {status['status']}")
        self.log_message(f"EPICS test: {status}")
    
    def check_queue_status(self):
        """Check and display queue status"""
        status = self.queue_manager.get_queue_status()
        self.queue_status_label.setText(f"Queue: {status['status']}")
        self.log_message(f"Queue status: {status}")
    
    def on_epics_connection_changed(self, connected):
        """Handle EPICS connection status changes"""
        status = "Connected" if connected else "Disconnected"
        self.epics_status_label.setText(f"EPICS: {status}")
        self.log_message(f"EPICS connection: {status}")
    
    def on_scan_started(self, scan_id):
        """Handle scan started event"""
        print(f"[GUI] Scan {scan_id} started")
        self.scan_id_label.setText(f"Current Scan: #{scan_id}")
        self.log_message(f"Coarse scan {scan_id} started")
    
    def on_scan_completed(self, scan_id):
        """Handle scan completed event"""
        print(f"[GUI] Scan {scan_id} completed")
        self.log_message(f"Scan {scan_id} completed")
    
    def on_zoom_scans_created(self, zoom_plans):
        """Handle zoom scans created event"""
        print(f"[GUI] {len(zoom_plans)} zoom scans created")
        self.zoom_label.setText(f"Zoom Scans: {len(zoom_plans)}")
        
        # Add to list
        for plan in zoom_plans:
            temp_id = plan.get('id', 'N/A')
            self.zoom_list.addItem(f"Zoom Scan {temp_id} (Region {plan['region_index']+1})")
        
        self.log_message(f"Created {len(zoom_plans)} zoom scan plans")
    
    def on_zoom_scans_submitted(self, submitted_plans):
        """Handle zoom scans submitted to queue"""
        print(f"[GUI] {len(submitted_plans)} zoom scans submitted to queue")
        self.submitted_label.setText(f"Submitted: {len(submitted_plans)}")
        self.log_message(f"Submitted {len(submitted_plans)} zoom scans to Bluesky queue")
    
    def on_control_message(self, message):
        """Handle control messages"""
        print(f"[GUI] Control message: {message}")
        self.log_message(f"Control: {message['type']}")
    
    def log_message(self, message):
        """Add message to log display"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")


def main():
    print(f"[MAIN] Starting auto-map ZMQ application...")
    
    # Create directories
    print(f"[MAIN] Creating directories...")
    TIFF_DIR.mkdir(exist_ok=True)
    SUBMIT_DIR.mkdir(exist_ok=True)
    print(f"[MAIN] Directories created successfully")
    
    # Start Qt application
    print(f"[MAIN] Initializing Qt application...")
    app = QApplication(sys.argv)
    
    # Create and show GUI
    gui = MicroscopeGUI()
    gui.show()
    
    print(f"[MAIN] GUI displayed, entering event loop...")
    
    # Run application
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print(f"[MAIN] Application interrupted by user")
    finally:
        print(f"[MAIN] Application shutting down...")


if __name__ == "__main__":
    main() 