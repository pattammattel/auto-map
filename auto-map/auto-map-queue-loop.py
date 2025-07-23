#!/usr/bin/env python3
"""
Simplified Autonomous Microscope Controller with Queue-Based Workflow
- Submit coarse scan to queue server
- Wait for completion
- Analyze data in the session
- Submit zoom scans to queue
- Wait for completion
- Repeat loop
"""

import sys
import os
import json
import zmq
import threading
import time
import numpy as np
import signal
import argparse
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread

# Configuration
TIFF_DIR = Path("./tiff_dir")
SUBMIT_DIR = Path("./submitted")
ZOOM_STEP_SIZE = 0.1

# ZMQ Configuration
ANALYSIS_PORT = 5556

# EPICS Configuration
DEFAULT_SCAN_PV = "SCAN:ID"

print(f"[INIT] Queue-based auto-map system starting...")
print(f"[INIT] TIFF directory: {TIFF_DIR}")
print(f"[INIT] Submit directory: {SUBMIT_DIR}")
print(f"[INIT] Analysis port: {ANALYSIS_PORT}")
print(f"[INIT] EPICS scan PV: {DEFAULT_SCAN_PV}")


class EPICSManager(QObject):
    """Manages EPICS integration for scan ID synchronization"""
    
    scan_id_updated = pyqtSignal(int)
    scan_status_updated = pyqtSignal(str)
    epics_connected = pyqtSignal(bool)
    
    def __init__(self, scan_pv_name=DEFAULT_SCAN_PV, status_pv_name="SCAN:STATUS"):
        super().__init__()
        self.scan_pv_name = scan_pv_name
        self.status_pv_name = status_pv_name
        self.current_scan_id = 0
        self.current_scan_status = "idle"
        self.connected = False
        self.monitoring = False
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_epics_values)
        
        print(f"[EPICS] EPICS manager initialized with PVs: {scan_pv_name}, {status_pv_name}")
    
    def start_monitoring(self):
        """Start monitoring EPICS scan ID and status"""
        self.monitoring = True
        self.monitor_timer.start(1000)  # Check every second
        print(f"[EPICS] Started monitoring scan ID and status")
    
    def stop_monitoring(self):
        """Stop monitoring EPICS scan ID and status"""
        self.monitoring = False
        self.monitor_timer.stop()
        print(f"[EPICS] Stopped monitoring scan ID and status")
    
    def check_epics_values(self):
        """Check current scan ID and status from EPICS"""
        try:
            # In real implementation, this would use epics.caget()
            # scan_id = epics.caget(self.scan_pv_name)
            # scan_status_bool = epics.caget(self.status_pv_name)  # Boolean value
            
            # Simulate EPICS values for demonstration
            import random
            new_scan_id = random.randint(1000, 9999)  # Simulated scan ID
            
            # Simulate boolean scan status (True = running, False = idle/completed)
            new_scan_status_bool = random.choice([True, False])
            
            # Convert boolean to status string
            if new_scan_status_bool:
                new_scan_status = "running"
            else:
                # If status was previously running and now False, it completed
                if self.current_scan_status == "running":
                    new_scan_status = "completed"
                else:
                    new_scan_status = "idle"
            
            # Update scan ID if changed
            if new_scan_id != self.current_scan_id:
                self.current_scan_id = new_scan_id
                self.scan_id_updated.emit(new_scan_id)
                print(f"[EPICS] Scan ID updated: {new_scan_id}")
            
            # Update scan status if changed
            if new_scan_status != self.current_scan_status:
                self.current_scan_status = new_scan_status
                self.scan_status_updated.emit(new_scan_status)
                print(f"[EPICS] Scan status updated: {new_scan_status} (boolean: {new_scan_status_bool})")
                
        except Exception as e:
            print(f"[EPICS] Error checking EPICS values: {e}")
            self.connected = False
            self.epics_connected.emit(False)
    
    def get_current_scan_id(self):
        """Get current scan ID"""
        return self.current_scan_id
    
    def get_current_scan_status(self):
        """Get current scan status"""
        return self.current_scan_status
    
    def get_current_scan_status_bool(self):
        """Get current scan status as boolean (True = running, False = idle/completed)"""
        return self.current_scan_status == "running"
    
    def set_scan_pv(self, pv_name):
        """Set scan PV name"""
        self.scan_pv_name = pv_name
        print(f"[EPICS] Scan PV set to: {pv_name}")
    
    def set_status_pv(self, pv_name):
        """Set scan status PV name"""
        self.status_pv_name = pv_name
        print(f"[EPICS] Status PV set to: {pv_name}")
    
    def test_connection(self):
        """Test EPICS connection"""
        try:
            # In real implementation, this would test both PVs
            # epics.caget(self.scan_pv_name)
            # epics.caget(self.status_pv_name)
            
            # Simulate connection test
            self.connected = True
            self.epics_connected.emit(True)
            print(f"[EPICS] Connection test successful")
            return True
        except Exception as e:
            self.connected = False
            self.epics_connected.emit(False)
            print(f"[EPICS] Connection test failed: {e}")
            return False


class BlueskyQueueManager(QObject):
    """Manages Bluesky queue server integration"""
    
    scan_submitted = pyqtSignal(int, str)
    scan_failed = pyqtSignal(int, str)
    scan_completed = pyqtSignal(int, str)
    
    def __init__(self):
        super().__init__()
        self.queue_connected = False
        self.submitted_scans = []
        self.completed_scans = []
        
        # Try to connect to Bluesky queue server
        try:
            # In real implementation, this would import and connect to Bluesky
            # import bluesky_queueserver_api
            # self.queue_connected = True
            print(f"[BLUESKY] Queue server connection simulated")
            self.queue_connected = True
        except ImportError:
            print(f"[BLUESKY] Bluesky queue server not available, using simulation mode")
            self.queue_connected = False
    
    def submit_coarse_scan(self, scan_plan):
        """Submit coarse scan to queue"""
        try:
            if not self.queue_connected:
                print(f"[BLUESKY] Simulating coarse scan submission: {scan_plan['id']}")
                self.submitted_scans.append({
                    "id": scan_plan['id'],
                    "type": "coarse",
                    "plan": scan_plan,
                    "status": "submitted",
                    "timestamp": time.time()
                })
                self.scan_submitted.emit(scan_plan['id'], f"coarse_scan_{scan_plan['id']}")
                return True
            
            # Real implementation would use Bluesky queue server API
            print(f"[BLUESKY] Submitting coarse scan to queue: {scan_plan['id']}")
            self.scan_submitted.emit(scan_plan['id'], f"coarse_scan_{scan_plan['id']}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to submit coarse scan {scan_plan['id']}: {str(e)}"
            print(f"[BLUESKY] {error_msg}")
            self.scan_failed.emit(scan_plan['id'], error_msg)
            return False
    
    def submit_zoom_scans(self, zoom_plans):
        """Submit zoom scans to queue"""
        try:
            if not self.queue_connected:
                print(f"[BLUESKY] Simulating zoom scan submission: {len(zoom_plans)} scans")
                for plan in zoom_plans:
                    self.submitted_scans.append({
                        "id": plan['id'],
                        "type": "zoom",
                        "plan": plan,
                        "status": "submitted",
                        "timestamp": time.time()
                    })
                    self.scan_submitted.emit(plan['id'], f"zoom_scan_{plan['id']}")
                return True
            
            # Real implementation would use Bluesky queue server API
            print(f"[BLUESKY] Submitting {len(zoom_plans)} zoom scans to queue")
            for plan in zoom_plans:
                self.scan_submitted.emit(plan['id'], f"zoom_scan_{plan['id']}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to submit zoom scans: {str(e)}"
            print(f"[BLUESKY] {error_msg}")
            self.scan_failed.emit(0, error_msg)
            return False
    
    def wait_for_scan_completion(self, scan_id, timeout=300):
        """Wait for a specific scan to complete"""
        print(f"[BLUESKY] Waiting for scan {scan_id} to complete...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if scan is completed
            if self.is_scan_completed(scan_id):
                print(f"[BLUESKY] Scan {scan_id} completed")
                self.scan_completed.emit(scan_id, f"scan_{scan_id}")
                return True
            
            time.sleep(1)  # Check every second
        
        print(f"[BLUESKY] Timeout waiting for scan {scan_id}")
        return False
    
    def is_scan_completed(self, scan_id):
        """Check if a scan is completed using EPICS status"""
        # In real implementation, this would check the queue server status
        # For now, we'll use EPICS status monitoring
        for scan in self.submitted_scans:
            if scan['id'] == scan_id and scan['status'] == 'submitted':
                # Check if scan has been running for a reasonable time
                if time.time() - scan['timestamp'] > 2:  # Minimum 2 seconds
                    # Mark as completed if EPICS status indicates completion
                    # This will be updated by the controller based on EPICS status
                    return False  # Let EPICS status determine completion
        return False
    
    def get_queue_status(self):
        """Get queue status"""
        return {
            "connected": self.queue_connected,
            "submitted": len(self.submitted_scans),
            "completed": len(self.completed_scans),
            "pending": len([s for s in self.submitted_scans if s['status'] == 'submitted'])
        }


class ZMQMessageHandler(QObject):
    """Handles ZMQ communication for analysis"""
    
    def __init__(self):
        super().__init__()
        self.context = zmq.Context()
        self.analysis_socket = self.context.socket(zmq.REQ)
        self.analysis_socket.connect(f"tcp://localhost:{ANALYSIS_PORT}")
        
        print(f"[ZMQ] Message handler initialized")
    
    def send_tiff_data_for_analysis(self, tiff_array, metadata):
        """Send TIFF array and metadata to analysis service"""
        try:
            # Prepare the analysis request
            request = {
                "type": "analyze_tiff_array",
                "tiff_array": tiff_array.tolist(),
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            print(f"[ZMQ] Sending TIFF array for analysis (shape: {tiff_array.shape})")
            
            # Send the request
            self.analysis_socket.send_json(request)
            
            # Wait for response
            response = self.analysis_socket.recv_json()
            print(f"[ZMQ] Analysis response received")
            return response
            
        except Exception as e:
            print(f"[ZMQ] Error sending TIFF data for analysis: {e}")
            return None
    
    def close(self):
        """Close ZMQ connections"""
        self.analysis_socket.close()


class AnalysisService(QThread):
    """Separate analysis service that processes TIFF arrays"""
    
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
                
            except zmq.error.Again:
                # No message available, continue
                time.sleep(0.1)
            except Exception as e:
                print(f"[ANALYSIS] Error in analysis service: {e}")
                time.sleep(1)
    
    def stop(self):
        """Stop the analysis service"""
        self.running = False
        self.socket.close()
        print(f"[ANALYSIS] Analysis service stopped")
    
    def analyze_tiff_array(self, tiff_array, metadata):
        """Analyze TIFF array and return regions of interest"""
        print(f"[ANALYSIS] Analyzing TIFF array with shape: {tiff_array.shape}")
        
        # Convert list back to numpy array
        tiff_array = np.array(tiff_array)
        
        # Simulated analysis - in real implementation this would process the TIFF array
        time.sleep(1)  # Simulate processing time
        
        # Generate regions based on array analysis
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
                    
                    regions.append({
                        "x": float(x_idx),
                        "y": float(y_idx),
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


class QueueBasedController(QObject):
    """Main controller that manages the queue-based workflow"""
    
    coarse_scan_started = pyqtSignal(int)
    coarse_scan_completed = pyqtSignal(int)
    zoom_scans_submitted = pyqtSignal(list)
    zoom_scans_completed = pyqtSignal(list)
    loop_completed = pyqtSignal(int)
    
    def __init__(self, epics_manager, queue_manager, message_handler):
        super().__init__()
        self.epics_manager = epics_manager
        self.queue_manager = queue_manager
        self.message_handler = message_handler
        self.running = False
        self.current_scan_id = 0
        self.current_scan_status = "idle"
        self.loop_count = 0
        self.stats = {
            "coarse_scans": 0,
            "zoom_scans": 0,
            "completed_loops": 0
        }
        
        # Connect signals
        self.queue_manager.scan_completed.connect(self.on_scan_completed)
        self.epics_manager.scan_status_updated.connect(self.on_scan_status_updated)
        
        print(f"[CONTROLLER] Queue-based controller initialized")
    
    def start(self):
        """Start the autonomous loop"""
        self.running = True
        self.loop_count = 0
        print(f"[CONTROLLER] Starting autonomous loop...")
        QTimer.singleShot(0, self.run_loop)
    
    def stop(self):
        """Stop the autonomous loop"""
        self.running = False
        print(f"[CONTROLLER] Stopping autonomous loop...")
    
    def run_loop(self):
        """Main loop: coarse scan → wait → analyze → zoom scans → wait → repeat"""
        if not self.running:
            return
        
        self.loop_count += 1
        print(f"[CONTROLLER] Starting loop {self.loop_count}")
        
        # Step 1: Submit coarse scan
        self.submit_coarse_scan()
    
    def submit_coarse_scan(self):
        """Submit coarse scan to queue"""
        print(f"[CONTROLLER] Submitting coarse scan...")
        
        # Get scan ID from EPICS
        self.current_scan_id = self.epics_manager.get_current_scan_id()
        
        # Create coarse scan plan
        coarse_plan = {
            "id": self.current_scan_id,
            "type": "coarse",
            "x_range": [0, 10],
            "y_range": [0, 10],
            "step_size": 0.5,
            "timestamp": time.time()
        }
        
        # Submit to queue
        if self.queue_manager.submit_coarse_scan(coarse_plan):
            self.coarse_scan_started.emit(self.current_scan_id)
            self.stats["coarse_scans"] += 1
            
            # Wait for completion
            QTimer.singleShot(0, lambda: self.wait_for_coarse_completion())
        else:
            print(f"[CONTROLLER] Failed to submit coarse scan")
            # Retry after delay
            QTimer.singleShot(5000, self.run_loop)
    
    def wait_for_coarse_completion(self):
        """Wait for coarse scan to complete using EPICS status"""
        print(f"[CONTROLLER] Waiting for coarse scan {self.current_scan_id} to complete...")
        print(f"[CONTROLLER] Current scan status: {self.current_scan_status}")
        
        # Check if scan is completed based on EPICS status
        if self.current_scan_status == "completed":
            self.on_coarse_scan_completed()
        elif self.current_scan_status == "error":
            print(f"[CONTROLLER] Coarse scan failed with error status")
            # Retry the loop
            QTimer.singleShot(5000, self.run_loop)
        else:
            # Check again in 1 second
            QTimer.singleShot(1000, self.wait_for_coarse_completion)
    
    def on_coarse_scan_completed(self):
        """Handle coarse scan completion"""
        print(f"[CONTROLLER] Coarse scan {self.current_scan_id} completed")
        self.coarse_scan_completed.emit(self.current_scan_id)
        
        # Step 2: Analyze data
        self.analyze_coarse_data()
    
    def analyze_coarse_data(self):
        """Analyze coarse scan data"""
        print(f"[CONTROLLER] Analyzing coarse scan data...")
        
        # Simulate TIFF array from coarse scan
        # In real implementation, this would load the actual TIFF data
        tiff_array = np.random.rand(100, 100)  # Simulated 100x100 array
        metadata = {
            "scan_id": self.current_scan_id,
            "scan_type": "coarse",
            "timestamp": time.time()
        }
        
        # Send for analysis via ZMQ
        analysis_result = self.message_handler.send_tiff_data_for_analysis(tiff_array, metadata)
        
        if analysis_result and analysis_result.get("regions"):
            regions = analysis_result["regions"]
            print(f"[CONTROLLER] Analysis found {len(regions)} regions")
            self.create_zoom_scans(regions)
        else:
            print(f"[CONTROLLER] No regions found, skipping zoom scans")
            # Complete loop without zoom scans
            self.complete_loop()
    
    def create_zoom_scans(self, regions):
        """Create and submit zoom scans"""
        print(f"[CONTROLLER] Creating zoom scans for {len(regions)} regions...")
        
        zoom_plans = []
        for i, region in enumerate(regions):
            zoom_plan = {
                "id": f"zoom_{self.current_scan_id}_{i+1}",
                "type": "zoom",
                "region_index": i,
                "x_start": region["x"] - region["w"] / 2,
                "x_end": region["x"] + region["w"] / 2,
                "y_start": region["y"] - region["h"] / 2,
                "y_end": region["y"] + region["h"] / 2,
                "step_size": ZOOM_STEP_SIZE,
                "confidence": region.get("confidence", 0.0),
                "timestamp": time.time()
            }
            zoom_plans.append(zoom_plan)
        
        # Submit zoom scans to queue
        if self.queue_manager.submit_zoom_scans(zoom_plans):
            self.zoom_scans_submitted.emit(zoom_plans)
            self.stats["zoom_scans"] += len(zoom_plans)
            
            # Wait for zoom scans to complete
            self.wait_for_zoom_completion(zoom_plans)
        else:
            print(f"[CONTROLLER] Failed to submit zoom scans")
            self.complete_loop()
    
    def wait_for_zoom_completion(self, zoom_plans):
        """Wait for all zoom scans to complete using EPICS status"""
        print(f"[CONTROLLER] Waiting for {len(zoom_plans)} zoom scans to complete...")
        print(f"[CONTROLLER] Current scan status: {self.current_scan_status}")
        
        # Check if zoom scans are completed based on EPICS status
        if self.current_scan_status == "completed":
            self.on_zoom_scans_completed(zoom_plans)
        elif self.current_scan_status == "error":
            print(f"[CONTROLLER] Zoom scans failed with error status")
            # Complete loop and continue
            self.complete_loop()
        else:
            # Check again in 1 second
            QTimer.singleShot(1000, lambda: self.wait_for_zoom_completion(zoom_plans))
    
    def on_zoom_scans_completed(self, zoom_plans):
        """Handle zoom scans completion"""
        print(f"[CONTROLLER] All zoom scans completed")
        self.zoom_scans_completed.emit(zoom_plans)
        self.complete_loop()
    
    def complete_loop(self):
        """Complete the current loop and start the next one"""
        print(f"[CONTROLLER] Loop {self.loop_count} completed")
        self.stats["completed_loops"] += 1
        self.loop_completed.emit(self.loop_count)
        
        # Start next loop after delay
        QTimer.singleShot(2000, self.run_loop)
    
    def on_scan_completed(self, scan_id, label):
        """Handle scan completion from queue manager"""
        print(f"[CONTROLLER] Scan completed: {label}")
    
    def on_scan_status_updated(self, new_status):
        """Handle EPICS scan status updates"""
        print(f"[CONTROLLER] Scan status updated: {new_status}")
        self.current_scan_status = new_status
        
        # Update scan status in queue manager
        for scan in self.queue_manager.submitted_scans:
            if scan['status'] == 'submitted':
                if new_status == "completed":
                    scan['status'] = 'completed'
                    self.queue_manager.completed_scans.append(scan)
                elif new_status == "error":
                    scan['status'] = 'error'
    
    def get_stats(self):
        """Get current statistics"""
        return self.stats.copy()


class QueueBasedMicroscopeController:
    """Main controller class for the queue-based system"""
    
    def __init__(self, epics_pv=DEFAULT_SCAN_PV, status_pv="SCAN:STATUS", auto_start=True):
        print(f"[MAIN] Initializing queue-based microscope controller...")
        
        # Create directories
        TIFF_DIR.mkdir(exist_ok=True)
        SUBMIT_DIR.mkdir(exist_ok=True)
        
        # Initialize components
        self.epics_manager = EPICSManager(epics_pv, status_pv)
        self.queue_manager = BlueskyQueueManager()
        self.message_handler = ZMQMessageHandler()
        self.analysis_service = AnalysisService()
        self.controller = QueueBasedController(
            self.epics_manager, 
            self.queue_manager, 
            self.message_handler
        )
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start components
        self.epics_manager.start_monitoring()
        self.analysis_service.start()
        
        if auto_start:
            QTimer.singleShot(1000, self.controller.start)
        
        print(f"[MAIN] Queue-based microscope controller initialized")
    
    def start(self):
        """Start the system"""
        print(f"[MAIN] Starting queue-based microscope system...")
        self.controller.start()
    
    def stop(self):
        """Stop the system"""
        print(f"[MAIN] Stopping queue-based microscope system...")
        self.controller.stop()
        self.analysis_service.stop()
        self.epics_manager.stop_monitoring()
        self.message_handler.close()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[MAIN] Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def get_status(self):
        """Get system status"""
        queue_status = self.queue_manager.get_queue_status()
        controller_stats = self.controller.get_stats()
        
        return {
            "epics_connected": self.epics_manager.connected,
            "current_scan_id": self.epics_manager.get_current_scan_id(),
            "current_scan_status": self.epics_manager.get_current_scan_status(),
            "current_scan_status_bool": self.epics_manager.get_current_scan_status_bool(),
            "queue_status": queue_status,
            "controller_stats": controller_stats,
            "loop_count": self.controller.loop_count,
            "running": self.controller.running
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        print(f"\n[STATUS] Queue-Based Microscope System Status:")
        print(f"  EPICS Connected: {status['epics_connected']}")
        print(f"  Current Scan ID: {status['current_scan_id']}")
        print(f"  Current Scan Status: {status['current_scan_status']} (boolean: {status['current_scan_status_bool']})")
        print(f"  Queue Connected: {status['queue_status']['connected']}")
        print(f"  Submitted Scans: {status['queue_status']['submitted']}")
        print(f"  Completed Scans: {status['queue_status']['completed']}")
        print(f"  Pending Scans: {status['queue_status']['pending']}")
        print(f"  Coarse Scans: {status['controller_stats']['coarse_scans']}")
        print(f"  Zoom Scans: {status['controller_stats']['zoom_scans']}")
        print(f"  Completed Loops: {status['controller_stats']['completed_loops']}")
        print(f"  Current Loop: {status['loop_count']}")
        print(f"  System Running: {status['running']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Queue-based Autonomous Microscope Controller")
    parser.add_argument("--epics-pv", default=DEFAULT_SCAN_PV, help="EPICS scan ID PV name")
    parser.add_argument("--status-pv", default="SCAN:STATUS", help="EPICS scan status PV name")
    parser.add_argument("--no-auto-start", action="store_true", help="Don't start automatically")
    parser.add_argument("--status-interval", type=int, default=30, help="Status print interval (seconds)")
    
    args = parser.parse_args()
    
    print(f"[MAIN] Queue-based auto-map system starting...")
    print(f"[MAIN] EPICS scan ID PV: {args.epics_pv}")
    print(f"[MAIN] EPICS scan status PV: {args.status_pv}")
    print(f"[MAIN] Auto-start: {not args.no_auto_start}")
    
    # Create controller
    controller = QueueBasedMicroscopeController(
        epics_pv=args.epics_pv,
        status_pv=args.status_pv,
        auto_start=not args.no_auto_start
    )
    
    # Status printing timer
    if args.status_interval > 0:
        status_timer = QTimer()
        status_timer.timeout.connect(controller.print_status)
        status_timer.start(args.status_interval * 1000)
    
    print(f"[MAIN] System started. Press Ctrl+C to stop.")
    
    # Keep the application running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[MAIN] Keyboard interrupt received")
        controller.stop()


if __name__ == "__main__":
    main() 