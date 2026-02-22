import time
import json
import os
import requests
from datetime import datetime

# =====================
# CONFIGURATION
# =====================

WEB_APP_URL = "https://script.google.com/macros/s/AKfycbwR3LH7qkeNNNZgEhOSMFqXZcO9xyVF7DiQau7gDxcTJ6ljtgD4EwrIm8tmC-B-fMpMag/exec"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Monitor all three files
FILES_TO_WATCH = {
    "fridge": os.path.join(SCRIPT_DIR, "fridge.json"),
    "weekly": os.path.join(SCRIPT_DIR, "weekly.json"),
    "monthly": os.path.join(SCRIPT_DIR, "monthly.json")
}

USERNAME = "admin"
PASSWORD = "password123"

# ---------------------
# CALCULATE STATUS
# ---------------------
def calc_status(expiry):
    if not expiry or expiry == "N/A":
        return "N/A", ""

    try:
        today = datetime.now().date()
        exp = datetime.strptime(expiry, "%Y-%m-%d").date()
        days_left = (exp - today).days

        if days_left < 0:
            return "Expired", days_left
        elif days_left <= 3: 
            return "Expired Soon", days_left
        else:
            return "Good to Eat", days_left
    except:
        return "N/A", ""

# ---------------------
# DATA PROCESSING
# ---------------------
def read_json_file(filepath):
    """Safely reads a JSON file and returns an empty list/dict if missing or broken."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return []

def flatten_fridge(data):
    """Converts the nested fridge dictionary into a flat list for Google Sheets."""
    flattened_items = []
    if isinstance(data, dict):
        for item_key, item_details in data.items():
            item_name = item_details.get("item_name", item_key)
            qty = item_details.get("qty", 0)
            unit = item_details.get("unit", "")
            category = item_details.get("category", "")
            expiry = item_details.get("expiry", "")
            status, days_left = calc_status(expiry)

            flattened_items.append({
                "item_name": item_name,
                "qty": qty,
                "unit": unit,
                "category": category,
                "expiry": expiry,
                "status": status,
                "days_left": days_left
            })
    return flattened_items

def push_to_google_sheet():
    """Reads all 3 files and uploads them in a single network request."""
    fridge_data = flatten_fridge(read_json_file(FILES_TO_WATCH["fridge"]))
    weekly_data = read_json_file(FILES_TO_WATCH["weekly"])
    monthly_data = read_json_file(FILES_TO_WATCH["monthly"])

    payload = {
        "username": USERNAME,
        "password": PASSWORD,
        "action": "update_all",  # New action tag so GAS knows what to do
        "fridge": fridge_data,
        "weekly": weekly_data,
        "monthly": monthly_data
    }

    try:
        print(">> Uploading to Google Sheets...")
        response = requests.post(WEB_APP_URL, json=payload)
        print(f">> Server Response: {response.text}")
    except Exception as e:
        print(f">> Error uploading: {e}")

# ---------------------
# WATCH FILES
# ---------------------
def main():
    print(f"*** Monitoring data files for changes ***")
    print("Keep this terminal open. Press Ctrl+C to stop.")

    # Initialize last modified times
    last_mtimes = {key: 0 for key in FILES_TO_WATCH}
    for key, path in FILES_TO_WATCH.items():
        if os.path.exists(path):
            last_mtimes[key] = os.path.getmtime(path)

    while True:
        try:
            time.sleep(1)
            changed = False

            # Check if any of the 3 files were updated
            for key, path in FILES_TO_WATCH.items():
                if os.path.exists(path):
                    current_mtime = os.path.getmtime(path)
                    if current_mtime != last_mtimes[key]:
                        changed = True
                        last_mtimes[key] = current_mtime

            # If any file changed, push the whole batch
            if changed:
                print(f"\n[Detected Change] Inventory data updated.")
                time.sleep(0.5) 
                push_to_google_sheet()

        except KeyboardInterrupt:
            print("\nStopping watcher...")
            break
        except Exception as e:
            print(f">> Loop Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()