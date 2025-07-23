#!/usr/bin/env python3
"""
Version Selection Helper for Autonomous Microscope Controller
Helps users choose between simulation and hardware versions.
"""

import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check which dependencies are available"""
    dependencies = {
        "epics": "pyepics",
        "bluesky": "bluesky_queueserver_api",
        "zmq": "zmq",
        "numpy": "numpy",
        "pyqt5": "PyQt5"
    }
    
    available = {}
    for name, package in dependencies.items():
        try:
            importlib.util.find_spec(package)
            available[name] = True
        except ImportError:
            available[name] = False
    
    return available

def print_version_info():
    """Print information about available versions"""
    print("=" * 80)
    print("Autonomous Microscope Controller - Version Selection")
    print("=" * 80)
    print()
    print("Available versions:")
    print()
    print("1. SIMULATION VERSION (auto-map-queue-loop-sim.py)")
    print("   - No hardware or EPICS required")
    print("   - Perfect for testing and development")
    print("   - Simulated scan timing and completion")
    print("   - All dependencies included")
    print()
    print("2. HARDWARE VERSION (auto-map-queue-loop-hw.py)")
    print("   - Requires real EPICS connections")
    print("   - Requires Bluesky queue server")
    print("   - Production-ready with real hardware")
    print("   - Needs pyepics and bluesky_queueserver_api")
    print()
    print("3. ORIGINAL VERSION (auto-map-queue-loop.py)")
    print("   - Hybrid version with simulation fallback")
    print("   - Good for development with optional hardware")
    print()

def check_requirements():
    """Check what's available and recommend a version"""
    print("Checking system requirements...")
    print()
    
    deps = check_dependencies()
    
    print("Available dependencies:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    print()
    
    # Determine recommended version
    if deps["epics"] and deps["bluesky"]:
        print("✓ RECOMMENDED: Hardware Version")
        print("   You have all required dependencies for production use.")
        print("   Run: python auto-map-queue-loop-hw.py")
        return "hw"
    elif deps["zmq"] and deps["numpy"] and deps["pyqt5"]:
        print("✓ RECOMMENDED: Simulation Version")
        print("   You have basic dependencies but missing EPICS/Bluesky.")
        print("   Run: python auto-map-queue-loop-sim.py")
        return "sim"
    else:
        print("✗ MISSING DEPENDENCIES")
        print("   Install required packages first:")
        print("   pip install pyzmq numpy PyQt5")
        return None

def install_dependencies():
    """Help install missing dependencies"""
    print("\nInstalling dependencies...")
    print()
    
    # Basic dependencies
    basic_deps = ["pyzmq", "numpy", "PyQt5"]
    print("Installing basic dependencies...")
    for dep in basic_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")
    
    print()
    print("For hardware version, also install:")
    print("  pip install pyepics bluesky-queueserver-api")
    print()

def run_version(version):
    """Run the selected version"""
    if version == "hw":
        print("Starting Hardware Version...")
        try:
            subprocess.run([sys.executable, "auto-map-queue-loop-hw.py"])
        except FileNotFoundError:
            print("Error: auto-map-queue-loop-hw.py not found")
    elif version == "sim":
        print("Starting Simulation Version...")
        try:
            subprocess.run([sys.executable, "auto-map-queue-loop-sim.py"])
        except FileNotFoundError:
            print("Error: auto-map-queue-loop-sim.py not found")
    elif version == "orig":
        print("Starting Original Version...")
        try:
            subprocess.run([sys.executable, "auto-map-queue-loop.py"])
        except FileNotFoundError:
            print("Error: auto-map-queue-loop.py not found")

def main():
    """Main function"""
    print_version_info()
    
    # Check requirements
    recommended = check_requirements()
    
    if recommended is None:
        print("Would you like to install basic dependencies? (y/n): ", end="")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            install_dependencies()
        return
    
    print()
    print("Choose a version to run:")
    print("1. Simulation Version (recommended for testing)")
    print("2. Hardware Version (requires EPICS/Bluesky)")
    print("3. Original Version (hybrid)")
    print("4. Install dependencies")
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                run_version("sim")
                break
            elif choice == "2":
                if recommended == "hw":
                    run_version("hw")
                else:
                    print("Hardware version requires EPICS and Bluesky dependencies.")
                    print("Install with: pip install pyepics bluesky-queueserver-api")
                break
            elif choice == "3":
                run_version("orig")
                break
            elif choice == "4":
                install_dependencies()
                print("Dependencies installed. Run this script again to choose a version.")
                break
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main() 