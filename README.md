# Autonomous Microscope Hardware Queue Controller

This project provides a flexible, hardware-connected queue-based controller for autonomous microscope scans. It is designed for both command-line and future GUI integration.

---

## Features
- **Loads scan parameters from a JSON file** (`controls.json` in the parent directory)
- **Submits scans to a Bluesky queue server**
- **Monitors scan status using EPICS PVs**
- **Performs custom analysis and generates zoom box JSON files**
- **Easily extensible for GUI use**

---

## Quick Start (Command Line)

1. **Install dependencies:**
   ```bash
   pip install pyepics bluesky-queueserver-api numpy
   ```

2. **Create a `controls.json` file in the parent directory:**
   ```json
   {
     "label": "my_scan",
     "det_names": ["det1", "det2"],
     "mot1": "x",
     "mot1_s": 0.0,
     "mot1_e": 10.0,
     "mot1_n": 101,
     "mot2": "y",
     "mot2_s": 0.0,
     "mot2_e": 5.0,
     "mot2_n": 51,
     "exp_t": 0.1
   }
   ```

3. **Run the controller:**
   ```bash
   python auto-map-queue-loop-hw.py --epics-pv SCAN:ID --status-pv SCAN:RUNNING
   ```
   - The script will load scan parameters from `controls.json` and submit the scan.
   - It will monitor scan status using the provided EPICS PVs.
   - After completion, it will perform analysis and generate zoom box JSON files.

---

## Configuration

- **Scan parameters** are loaded from `../controls.json` (relative to the script).
- **Required fields:**
  - `label`, `det_names`, `mot1`, `mot1_s`, `mot1_e`, `mot1_n`, `mot2`, `mot2_s`, `mot2_e`, `mot2_n`, `exp_t`
- **If any field is missing,** the script will print an error and retry.

---

## Best Practices for GUI Integration

- **Avoid argparse for core logic.**
  - Use argparse only in the `if __name__ == "__main__":` block for CLI entry.
- **Encapsulate all logic in classes/functions** that take parameters as arguments.
- **Let the GUI import your classes/functions** and call them directly, passing parameters as needed.
- **Example structure:**

  ```python
  # my_scan_module.py
  class SimpleMicroscopeController:
      def __init__(self, epics_pv, status_pv, controls_path):
          ...
      def start(self):
          ...

  # CLI entry point
  if __name__ == "__main__":
      import argparse
      parser = argparse.ArgumentParser()
      parser.add_argument("--epics-pv", ...)
      parser.add_argument("--status-pv", ...)
      parser.add_argument("--controls", ...)
      args = parser.parse_args()
      controller = SimpleMicroscopeController(
          epics_pv=args.epics_pv,
          status_pv=args.status_pv,
          controls_path=args.controls
      )
      controller.start()
  ```

- **For GUI:**
  ```python
  from my_scan_module import SimpleMicroscopeController
  controller = SimpleMicroscopeController(
      epics_pv="SCAN:ID",
      status_pv="SCAN:RUNNING",
      controls_path="path/to/controls.json"
  )
  controller.start()
  ```

---

## Customization
- **Edit `controls.json`** to change scan parameters without modifying code.
- **Add your custom analysis code** in the `CustomAnalysis` class.
- **Extend for zoom scans or other scan types** as needed.

---

## Troubleshooting
- **Missing fields in `controls.json`:** The script will print an error and retry.
- **EPICS or queue server not connected:** The script will print a warning and not start.
- **For GUI integration:** Refactor to keep all logic in importable classes/functions.

---

## License
MIT 