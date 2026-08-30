import os
import sys
from pathlib import Path

studio_src = Path("triune_studio/src").resolve()
index_html = studio_src / "index.html"

print(f"Index HTML path: {index_html}")
print(f"Index HTML exists: {index_html.exists()}")

# Read index.html content
print("=" * 50)
print(index_html.read_text(encoding="utf-8"))
print("=" * 50)

# Check vendor files
vendor_dir = studio_src / "vendor"
print("Vendor dir exists:", vendor_dir.exists())
if vendor_dir.exists():
    for f in vendor_dir.iterdir():
        print(f"  Vendor file: {f.name} ({f.stat().st_size} bytes)")
