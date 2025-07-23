#!/usr/bin/env python3
"""
Simulation-Only Queue-Based Autonomous Microscope Controller
- No EPICS dependencies
- Simulated scan status and completion
- Perfect for testing and development
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

print(f"[INIT] Simulation-only queue-based auto-map system starting...")
print(f"[INIT] TIFF directory: {TIFF_DIR}")
print(f"[INIT] Submit directory: {SUBMIT_DIR}")
print(f"[INIT] Analysis port: {ANALYSIS_PORT}")


class SimulatedEPICSManager(QObject):
    """Simulated EPICS manager for testing without hardware"""
    
    scan_id_updated = pyqtSignal(int)
    scan_status_updated = pyqtSignal(str)
    epics_connected = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.current_scan_id = 0
        self.current_scan_status = "idle"
        self.connected = True
        self.monitoring = False
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.simulate_epics_values)
        
        print(f"[SIM-EPICS] Simulated EPICS manager initialized")
    
    def start_monitoring(self):
        """Start monitoring simulated EPICS values"""
        self.monitoring = True
        self.monitor_timer.start(2000)  # Check every 2 seconds
        print(f"[SIM-EPICS] Started monitoring simulated scan ID and status")
    
    def stop_monitoring(self):
        """Stop monitoring simulated EPICS values"""
        self.monitoring = False
        self.monitor_timer.stop()
        print(f"[SIM-EPICS] Stopped monitoring")
    
    def simulate_epics_values(self):
        """Simulate EPICS scan ID and status changes"""
        import random
        
        # Simulate scan ID changes
        if random.random() < 0.3:  # 30% chance to change scan ID
            new_scan_id = random.randint(1000, 9999)
            if new_scan_id != self.current_scan_id:
                self.current_scan_id = new_scan_id
                self.scan_id_updated.emit(new_scan_id)
                print(f"[SIM-EPICS] Scan ID updated: {new_scan_id}")
        
        # Simulate status transitions
        if random.random() < 0.4:  # 40% chance to change status
            if self.current_scan_status == "idle":
                new_status = "running"
            elif self.current_scan_status == "running":
                new_status = "completed"
            elif self.current_scan_status == "completed":
                new_status = "idle"
            else:
                new_status = "idle"
            
            if new_status != self.current_scan_status:
                self.current_scan_status = new_status
                self.scan_status_updated.emit(new_status)
                print(f"[SIM-EPICS] Scan status updated: {new_status}")
    
    def get_current_scan_id(self):
        """Get current scan ID"""
        return self.current_scan_id
    
    def get_current_scan_status(self):
        """Get current scan status"""
        return self.current_scan_status
    
    def get_current_scan_status_bool(self):
        """Get current scan status as boolean (True = running, False = idle/completed)"""
        return self.current_scan_status == "running"
    
    def test_connection(self):
        """Test simulated EPICS connection"""
        self.connected = True
        self.epics_connected.emit(True)
        print(f"[SIM-EPICS] Connection test successful (simulated)")
        return True


class SimulatedQueueManager(QObject):
    """Simulated Bluesky queue manager for testing"""
    
    scan_submitted = pyqtSignal(int, str)
    scan_failed = pyqtSignal(int, str)
    scan_completed = pyqtSignal(int, str)
    
    def __init__(self):
        super().__init__()
        self.submitted_scans = []
        self.completed_scans = []
        self.scan_completion_times = {}  # Track when scans should complete
        
        print(f"[SIM-QUEUE] Simulated queue manager initialized")
    
    def submit_coarse_scan(self, scan_plan):
        """Submit coarse scan to simulated queue"""
        try:
            print(f"[SIM-QUEUE] Simulating coarse scan submission: {scan_plan['id']}")
            
            # Simulate scan completion after 3-8 seconds
            completion_delay = 3 + np.random.random() * 5
            completion_time = time.time() + completion_delay
            
            self.submitted_scans.append({
                "id": scan_plan['id'],
                "type": "coarse",
                "plan": scan_plan,
                "status": "submitted",
                "timestamp": time.time(),
                "completion_time": completion_time
            })
            
            self.scan_submitted.emit(scan_plan['id'], f"coarse_scan_{scan_plan['id']}")
            print(f"[SIM-QUEUE] Coarse scan {scan_plan['id']} will complete in {completion_delay:.1f} seconds")
            return True
            
        except Exception as e:
            error_msg = f"Failed to submit coarse scan {scan_plan['id']}: {str(e)}"
            print(f"[SIM-QUEUE] {error_msg}")
            self.scan_failed.emit(scan_plan['id'], error_msg)
            return False
    
    def submit_zoom_scans(self, zoom_plans):
        """Submit zoom scans to simulated queue"""
        try:
            print(f"[SIM-QUEUE] Simulating zoom scan submission: {len(zoom_plans)} scans")
            
            for plan in zoom_plans:
                # Simulate scan completion after 2-5 seconds
                completion_delay = 2 + np.random.random() * 3
                completion_time = time.time() + completion_delay
                
                self.submitted_scans.append({
                    "id": plan['id'],
                    "type": "zoom",
                    "plan": plan,
                    "status": "submitted",
                    "timestamp": time.time(),
                    "completion_time": completion_time
                })
                
                self.scan_submitted.emit(plan['id'], f"zoom_scan_{plan['id']}")
                print(f"[SIM-QUEUE] Zoom scan {plan['id']} will complete in {completion_delay:.1f} seconds")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to submit zoom scans: {str(e)}"
            print(f"[SIM-QUEUE] {error_msg}")
            self.scan_failed.emit(0, error_msg)
            return False
    
    def is_scan_completed(self, scan_id):
        """Check if a scan is completed based on simulated timing"""
        current_time = time.time()
        
        for scan in self.submitted_scans:
            if scan['id'] == scan_id and scan['status'] == 'submitted':
                if current_time >= scan['completion_time']:
                    scan['status'] = 'completed'
                    self.completed_scans.append(scan)
                    print(f"[SIM-QUEUE] Scan {scan_id} completed (simulated)")
                    return True
        return False
    
    def get_queue_status(self):
        """Get simulated queue status"""
        pending = len([s for s in self.submitted_scans if s['status'] == 'submitted'])
        return {
            "connected": True,
            "submitted": len(self.submitted_scans),
            "completed": len(self.completed_scans),
            "pending": pending
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


class SimulatedQueueController(QObject):
    """Main controller that manages the simulated queue-based workflow"""
    
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
        
        print(f"[SIM-CONTROLLER] Simulated queue-based controller initialized")
    
    def start(self):
        """Start the autonomous loop"""
        self.running = True
        self.loop_count = 0
        print(f"[SIM-CONTROLLER] Starting autonomous loop...")
        QTimer.singleShot(0, self.run_loop)
    
    def stop(self):
        """Stop the autonomous loop"""
        self.running = False
        print(f"[SIM-CONTROLLER] Stopping autonomous loop...")
    
    def run_loop(self):
        """Main loop: coarse scan → wait → analyze → zoom scans → wait → repeat"""
        if not self.running:
            return
        
        self.loop_count += 1
        print(f"[SIM-CONTROLLER] Starting loop {self.loop_count}")
        
        # Step 1: Submit coarse scan
        self.submit_coarse_scan()
    
    def submit_coarse_scan(self):
        """Submit coarse scan to queue"""
        print(f"[SIM-CONTROLLER] Submitting coarse scan...")
        
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
            print(f"[SIM-CONTROLLER] Failed to submit coarse scan")
            # Retry after delay
            QTimer.singleShot(5000, self.run_loop)
    
    def wait_for_coarse_completion(self):
        """Wait for coarse scan to complete using simulated timing"""
        print(f"[SIM-CONTROLLER] Waiting for coarse scan {self.current_scan_id} to complete...")
        
        # Check if scan is completed based on simulated timing
        if self.queue_manager.is_scan_completed(self.current_scan_id):
            self.on_coarse_scan_completed()
        else:
            # Check again in 1 second
            QTimer.singleShot(1000, self.wait_for_coarse_completion)
    
    def on_coarse_scan_completed(self):
        """Handle coarse scan completion"""
        print(f"[SIM-CONTROLLER] Coarse scan {self.current_scan_id} completed")
        self.coarse_scan_completed.emit(self.current_scan_id)
        
        # Step 2: Analyze data
        self.analyze_coarse_data()
    
    def analyze_coarse_data(self):
        """Analyze coarse scan data"""
        print(f"[SIM-CONTROLLER] Analyzing coarse scan data...")
        
        # Simulate TIFF array from coarse scan
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
            print(f"[SIM-CONTROLLER] Analysis found {len(regions)} regions")
            self.create_zoom_scans(regions)
        else:
            print(f"[SIM-CONTROLLER] No regions found, skipping zoom scans")
            # Complete loop without zoom scans
            self.complete_loop()
    
    def create_zoom_scans(self, regions):
        """Create and submit zoom scans"""
        print(f"[SIM-CONTROLLER] Creating zoom scans for {len(regions)} regions...")
        
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
            print(f"[SIM-CONTROLLER] Failed to submit zoom scans")
            self.complete_loop()
    
    def wait_for_zoom_completion(self, zoom_plans):
        """Wait for all zoom scans to complete using simulated timing"""
        print(f"[SIM-CONTROLLER] Waiting for {len(zoom_plans)} zoom scans to complete...")
        
        # Check if all zoom scans are completed based on simulated timing
        all_completed = True
        for plan in zoom_plans:
            if not self.queue_manager.is_scan_completed(plan["id"]):
                all_completed = False
                break
        
        if all_completed:
            self.on_zoom_scans_completed(zoom_plans)
        else:
            # Check again in 1 second
            QTimer.singleShot(1000, lambda: self.wait_for_zoom_completion(zoom_plans))
    
    def on_zoom_scans_completed(self, zoom_plans):
        """Handle zoom scans completion"""
        print(f"[SIM-CONTROLLER] All zoom scans completed")
        self.zoom_scans_completed.emit(zoom_plans)
        self.complete_loop()
    
    def complete_loop(self):
        """Complete the current loop and start the next one"""
        print(f"[SIM-CONTROLLER] Loop {self.loop_count} completed")
        self.stats["completed_loops"] += 1
        self.loop_completed.emit(self.loop_count)
        
        # Start next loop after delay
        QTimer.singleShot(2000, self.run_loop)
    
    def on_scan_completed(self, scan_id, label):
        """Handle scan completion from queue manager"""
        print(f"[SIM-CONTROLLER] Scan completed: {label}")
    
    def on_scan_status_updated(self, new_status):
        """Handle EPICS scan status updates"""
        print(f"[SIM-CONTROLLER] Scan status updated: {new_status}")
        self.current_scan_status = new_status
    
    def get_stats(self):
        """Get current statistics"""
        return self.stats.copy()


class SimulatedMicroscopeController:
    """Main controller class for the simulated queue-based system"""
    
    def __init__(self, auto_start=True):
        print(f"[MAIN] Initializing simulated queue-based microscope controller...")
        
        # Create directories
        TIFF_DIR.mkdir(exist_ok=True)
        SUBMIT_DIR.mkdir(exist_ok=True)
        
        # Initialize components
        self.epics_manager = SimulatedEPICSManager()
        self.queue_manager = SimulatedQueueManager()
        self.message_handler = ZMQMessageHandler()
        self.analysis_service = AnalysisService()
        self.controller = SimulatedQueueController(
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
        
        print(f"[MAIN] Simulated microscope controller initialized")
    
    def start(self):
        """Start the system"""
        print(f"[MAIN] Starting simulated microscope system...")
        self.controller.start()
    
    def stop(self):
        """Stop the system"""
        print(f"[MAIN] Stopping simulated microscope system...")
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
        print(f"\n[STATUS] Simulated Queue-Based Microscope System Status:")
        print(f"  EPICS Connected: {status['epics_connected']} (simulated)")
        print(f"  Current Scan ID: {status['current_scan_id']}")
        print(f"  Current Scan Status: {status['current_scan_status']} (boolean: {status['current_scan_status_bool']})")
        print(f"  Queue Connected: {status['queue_status']['connected']} (simulated)")
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
    parser = argparse.ArgumentParser(description="Simulated Queue-based Autonomous Microscope Controller")
    parser.add_argument("--no-auto-start", action="store_true", help="Don't start automatically")
    parser.add_argument("--status-interval", type=int, default=30, help="Status print interval (seconds)")
    
    args = parser.parse_args()
    
    print(f"[MAIN] Simulated queue-based auto-map system starting...")
    print(f"[MAIN] Auto-start: {not args.no_auto_start}")
    print(f"[MAIN] This is a SIMULATION - no real hardware or EPICS required")
    
    # Create controller
    controller = SimulatedMicroscopeController(auto_start=not args.no_auto_start)
    
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