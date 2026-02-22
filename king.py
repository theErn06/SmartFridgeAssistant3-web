import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# 1. Start the Watcher in the background
print("Starting Background Watcher...")
p_watcher = subprocess.Popen([sys.executable, os.path.join(BASE, "watcher.py")])

# 2. Start Jarvis in the foreground so it owns the keyboard input
try:
    subprocess.run([sys.executable, os.path.join(BASE, "jarvis.py")])
except KeyboardInterrupt:
    pass
finally:
    # 3. Cleanly shut down the watcher when you exit Jarvis
    print("\nShutting down Watcher...")
    p_watcher.terminate()