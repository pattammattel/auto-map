import sys
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QListWidget
)
from PyQt5.QtCore import (
    QObject, pyqtSignal, QTimer, QFileSystemWatcher
)

TIFF_DIR = Path("./tiff_dir")
SUBMIT_DIR = Path("./submitted")
ZOOM_STEP_SIZE = 0.1

print(f"[INIT] Auto-map system starting...")
print(f"[INIT] TIFF directory: {TIFF_DIR}")
print(f"[INIT] Submit directory: {SUBMIT_DIR}")
print(f"[INIT] Zoom step size: {ZOOM_STEP_SIZE}")


class AutoMicroscope(QObject):
    coarseScanStarted = pyqtSignal(int)
    zoomScanSubmitted = pyqtSignal(int)
    zoomScanCountUpdated = pyqtSignal(int)
    zoomScansCompleted = pyqtSignal()

    def __init__(self):
        super().__init__()
        print(f"[INIT] AutoMicroscope controller initialized")
        self.scan_id = 0
        self.zoom_phase = False

        self.fs_watcher = QFileSystemWatcher()
        self.fs_watcher.addPath(str(TIFF_DIR))
        self.fs_watcher.directoryChanged.connect(self.check_for_tiffs)
        print(f"[INIT] File system watcher set up for: {TIFF_DIR}")

    def start(self):
        print(f"[START] Starting autonomous microscope system...")
        QTimer.singleShot(0, self.send_coarse_scan)

    def check_for_tiffs(self):
        print(f"[WATCHER] Checking for new TIFF files in {TIFF_DIR}...")
        tiffs = sorted(f for f in os.listdir(TIFF_DIR) if f.endswith(".tiff"))
        print(f"[WATCHER] Found {len(tiffs)} TIFF files: {tiffs}")
        if tiffs:
            tiff_path = TIFF_DIR / tiffs[-1]
            print(f"[Watcher] New TIFF detected: {tiff_path}")
            self.handle_new_tiff(tiff_path)
        else:
            print(f"[WATCHER] No TIFF files found")

    def send_coarse_scan(self):
        print(f"[SCAN] Preparing to send coarse scan...")
        self.zoom_phase = False
        self.scan_id += 1
        plan = {"type": "coarse", "id": self.scan_id, "x": [0, 10], "y": [0, 10], "step": 0.5}
        print(f"[SCAN] Coarse scan plan created: {plan}")
        self.save_plan(plan)
        print(f"[SCAN] Coarse scan {self.scan_id} submitted.")
        self.coarseScanStarted.emit(self.scan_id)

    def handle_new_tiff(self, tiff_path):
        print(f"[HANDLER] Processing new TIFF: {tiff_path}")
        if not self.zoom_phase:
            print(f"[HANDLER] Not in zoom phase, analyzing TIFF...")
            regions = self.analyze_tiff(tiff_path)
            print(f"[HANDLER] Found {len(regions)} regions of interest")
            self.send_zoom_scans(regions)
            QTimer.singleShot(2000, self.on_zoom_scans_done)
        else:
            print(f"[HANDLER] Already in zoom phase, skipping TIFF processing")

    def analyze_tiff(self, path):
        print(f"[PROCESS] Analyzing {path}...")
        # Simulated analysis - in real implementation this would process the TIFF
        regions = [
            {"x": 10.0, "y": 5.0, "w": 1.0, "h": 1.0},
            {"x": 15.0, "y": 8.0, "w": 0.5, "h": 0.5}
        ]
        print(f"[PROCESS] Analysis complete. Regions: {regions}")
        return regions

    def send_zoom_scans(self, regions):
        print(f"[ZOOM] Starting zoom phase with {len(regions)} regions...")
        self.zoom_phase = True
        count = 0
        for i, region in enumerate(regions):
            self.scan_id += 1
            plan = {
                "type": "zoom",
                "id": self.scan_id,
                "x_start": region["x"] - region["w"] / 2,
                "x_end": region["x"] + region["w"] / 2,
                "y_start": region["y"] - region["h"] / 2,
                "y_end": region["y"] + region["h"] / 2,
                "step": ZOOM_STEP_SIZE
            }
            print(f"[ZOOM] Creating zoom scan {self.scan_id} for region {i+1}: {plan}")
            self.save_plan(plan)
            self.zoomScanSubmitted.emit(self.scan_id)
            count += 1
        print(f"[ZOOM] Created {count} zoom scan plans")
        self.zoomScanCountUpdated.emit(count)

    def on_zoom_scans_done(self):
        print("[STATUS] Zoom scans complete. Restarting coarse scan.")
        self.zoomScansCompleted.emit()
        self.send_coarse_scan()

    def save_plan(self, plan):
        print(f"[SAVE] Saving scan plan {plan['id']}...")
        SUBMIT_DIR.mkdir(exist_ok=True)
        out_file = SUBMIT_DIR / f"scan_{plan['id']}.json"
        with open(out_file, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"[SAVE] Plan saved to: {out_file}")


class ScanGUI(QWidget):
    def __init__(self):
        super().__init__()
        print(f"[GUI] Initializing GUI...")
        self.setWindowTitle("Autonomous Microscope Controller")
        self.resize(400, 300)

        self.coarse_label = QLabel("Coarse Scan: Not started")
        self.zoom_list = QListWidget()
        self.zoom_counter = QLabel("Zoom Scans: 0")
        self.status = QLabel("Status: Idle")
        self.start_button = QPushButton("Start System")

        layout = QVBoxLayout()
        layout.addWidget(self.coarse_label)
        layout.addWidget(self.zoom_counter)
        layout.addWidget(QLabel("Zoom Scan IDs:"))
        layout.addWidget(self.zoom_list)
        layout.addWidget(self.status)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.controller = AutoMicroscope()
        self.controller.coarseScanStarted.connect(self.on_coarse_started)
        self.controller.zoomScanSubmitted.connect(self.on_zoom_scan_submitted)
        self.controller.zoomScanCountUpdated.connect(self.on_zoom_count_updated)
        self.controller.zoomScansCompleted.connect(self.on_zoom_done)

        self.start_button.clicked.connect(self.controller.start)
        print(f"[GUI] GUI initialization complete")

    def on_coarse_started(self, scan_id):
        print(f"[GUI] Coarse scan {scan_id} started")
        self.coarse_label.setText(f"Coarse Scan: #{scan_id}")
        self.status.setText("Status: Coarse scan running...")

    def on_zoom_scan_submitted(self, scan_id):
        print(f"[GUI] Zoom scan {scan_id} submitted")
        self.zoom_list.addItem(f"Zoom Scan #{scan_id}")

    def on_zoom_count_updated(self, count):
        print(f"[GUI] Zoom scan count updated: {count}")
        self.zoom_counter.setText(f"Zoom Scans: {count}")
        self.status.setText("Status: Zoom scans submitted")

    def on_zoom_done(self):
        print(f"[GUI] Zoom scans completed")
        self.status.setText("Status: Zoom complete, starting next coarse scan")


def main():
    print(f"[MAIN] Starting auto-map application...")
    print(f"[MAIN] Creating directories...")
    TIFF_DIR.mkdir(exist_ok=True)
    SUBMIT_DIR.mkdir(exist_ok=True)
    print(f"[MAIN] Directories created successfully")
    
    print(f"[MAIN] Initializing Qt application...")
    app = QApplication(sys.argv)
    gui = ScanGUI()
    gui.show()
    print(f"[MAIN] GUI displayed, entering event loop...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()