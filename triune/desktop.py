"""Native Windows Desktop Application Launcher for Triune Studio."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser


def launch_desktop_app(port: int = 8000) -> None:
    """Launch embedded Triune Studio API server and open as a native desktop application window."""
    from triune.api import run_server

    print(f"🚀 Starting Triune Framework Engine on http://127.0.0.1:{port}")
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "127.0.0.1", "port": port},
        daemon=True,
    )
    server_thread.start()

    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}/"

    # 1. Try PyWebView for a pure native desktop window
    try:
        import webview

        print("📱 Launching Triune Studio Native Window via PyWebView...")
        webview.create_window(
            title="Triune Studio – AI Engine & Research IDE",
            url=url,
            width=1340,
            height=880,
            resizable=True,
            min_size=(900, 600),
        )
        webview.start()
        return
    except ImportError:
        pass

    # 2. Try MS Edge Native Windows App Mode (--app=http://127.0.0.1:8000/)
    if sys.platform == "win32":
        try:
            print("📱 Launching Triune Studio Native Windows App Mode (MS Edge WebView)...")
            edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            if not os.path.exists(edge_path):
                edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

            if os.path.exists(edge_path):
                subprocess.Popen([edge_path, f"--app={url}", "--name=Triune Studio"])
                return
        except Exception:
            pass

    # 3. Fallback to default browser
    print("🌐 Opening Triune Studio in browser window...")
    webbrowser.open(url)
