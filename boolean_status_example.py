#!/usr/bin/env python3
"""
Example script demonstrating the queue-based autonomous microscope loop
with boolean scan status PV monitoring.
"""

import sys
import time
from auto_map_queue_loop import QueueBasedMicroscopeController

def main():
    """Demonstrate the queue-based loop system with boolean status PV"""
    print("=" * 70)
    print("Queue-Based Autonomous Microscope Loop with Boolean Status PV")
    print("=" * 70)
    print()
    print("This example demonstrates the complete workflow with boolean status:")
    print("1. Submit coarse scan to queue server")
    print("2. Monitor boolean status PV (True = running, False = idle/completed)")
    print("3. Wait for status to transition from True to False (completion)")
    print("4. Analyze data in the session")
    print("5. Submit zoom scans to queue")
    print("6. Wait for zoom scan completion (status True → False)")
    print("7. Repeat the loop")
    print()
    print("Boolean Status PV Interpretation:")
    print("  True  = Scan is running")
    print("  False = Scan is idle or completed")
    print("  Transition from True to False = Scan completed")
    print()
    
    # Create the controller with boolean status PV
    print("[EXAMPLE] Creating queue-based microscope controller...")
    controller = QueueBasedMicroscopeController(
        epics_pv="SCAN:ID",
        status_pv="SCAN:RUNNING",  # Boolean status PV
        auto_start=False  # Don't start automatically
    )
    
    # Print initial status
    print("\n[EXAMPLE] Initial system status:")
    controller.print_status()
    
    # Start the system
    print("\n[EXAMPLE] Starting the autonomous loop...")
    controller.start()
    
    # Monitor the system for a few loops
    print("\n[EXAMPLE] Monitoring system for 60 seconds...")
    print("Press Ctrl+C to stop early")
    print()
    
    try:
        for i in range(60):  # Monitor for 60 seconds
            time.sleep(1)
            
            # Print status every 10 seconds
            if (i + 1) % 10 == 0:
                print(f"\n[EXAMPLE] Status update at {i+1} seconds:")
                controller.print_status()
                
    except KeyboardInterrupt:
        print("\n[EXAMPLE] Keyboard interrupt received")
    
    # Stop the system
    print("\n[EXAMPLE] Stopping the system...")
    controller.stop()
    
    # Final status
    print("\n[EXAMPLE] Final system status:")
    controller.print_status()
    
    print("\n[EXAMPLE] Example completed!")
    print("\n[EXAMPLE] Key points about boolean status PV:")
    print("  - True = scan is actively running")
    print("  - False = scan is idle or has completed")
    print("  - System detects completion when status changes from True to False")
    print("  - This provides reliable scan completion detection")

if __name__ == "__main__":
    main() 