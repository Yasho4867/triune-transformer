"""Native Windows Desktop Launcher for Triune Studio."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triune.desktop import launch_desktop_app

if __name__ == "__main__":
    launch_desktop_app(port=8000)
