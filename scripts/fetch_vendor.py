"""Download untruncated vendor JS assets for React 18 & ReactDOM 18."""

import urllib.request
from pathlib import Path

studio_vendor = Path("triune_studio/src/vendor").resolve()
studio_vendor.mkdir(parents=True, exist_ok=True)

root_vendor = Path("studio/vendor").resolve()
root_vendor.mkdir(parents=True, exist_ok=True)

urls = {
    "react.production.min.js": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom.production.min.js": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    "babel.min.js": "https://unpkg.com/@babel/standalone/babel.min.js"
}

for filename, url in urls.items():
    print(f"Downloading {filename} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
    
    target_path = studio_vendor / filename
    target_path.write_bytes(content)
    print(f"Saved {filename} ({len(content)} bytes) to {target_path}")

    root_path = root_vendor / filename
    root_path.write_bytes(content)
