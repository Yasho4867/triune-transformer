import signal
import sys

def sigint_handler(sig, frame):
    print("\n⚠️ KeyboardInterrupt – saving checkpoint...")
    save_latest(step, 0.0)
    sys.exit(0)

def sigterm_handler(sig, frame):
    print("\n⚠️ SIGTERM received – saving checkpoint...")
    save_latest(step, 0.0)
    sys.exit(0)

