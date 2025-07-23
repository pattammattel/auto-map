#!/usr/bin/env python3
"""
Example script demonstrating the queue-based autonomous microscope loop.
This shows the complete workflow: coarse scan → wait → analyze → zoom scans → wait → repeat.
"""

import sys
import time
from auto_map_queue_loop import QueueBasedMicroscopeController

def main():
    """Demonstrate the queue-based loop system"""
    print("=" * 60)
    print("Queue-Based Autonomous Microscope Loop Example")
    print("=" * 60)
    print()
    print("This example demonstrates the complete workflow:")
    print("1. Submit coarse scan to queue server")
    print("2. Wait for coarse scan completion")
    print("3. Analyze data in the session")
    print("4. Submit zoom scans to queue")
    print("5. Wait for zoom scan completion")
    print("6. Repeat the loop")
    print()
    
    # Create the controller
    print("[EXAMPLE] Creating queue-based microscope controller...")
    controller = QueueBasedMicroscopeController(
        epics_pv="SCAN:ID",
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

if __name__ == "__main__":
    main() 