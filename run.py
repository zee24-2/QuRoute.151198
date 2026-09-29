"""
One-click launcher for the QPSO-Powered Traffic Route Optimizer.
Starts the FastAPI server with Uvicorn and launches the default web browser.
"""

import sys
import os
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:8000"
    print(f"\n=======================================================")
    print(f"  QPSO Traffic Route Optimizer is LIVE at: {url}")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    # Ensure current directory is on PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")
