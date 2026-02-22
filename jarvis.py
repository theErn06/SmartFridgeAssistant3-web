#!/usr/bin/env python3
"""
Combined Jarvis Smart Fridge Assistant
- FEATURES: Section B's STT2 (Silero VAD + Faster Whisper), Clean JSON, Waste Tracking.
- MERGED: Section A's Arduino (Door/LCD), Wake Word, Door Watchdog, and Regex Number Fallback.
"""

import os
import json
import requests
import pyttsx3
import numpy as np
import sounddevice as sd
import torch
import queue
import sys
import time
import collections
import re
import serial
import threading
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from typing import List, Dict, Any

# ------------------------
# CONFIGURATION
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(SCRIPT_DIR, "fridge.json")
WASTE_FILE = os.path.join(SCRIPT_DIR, "waste_log.json")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "jarvis"

# ==========================================
# DYNAMIC NOISE SETTINGS (HIGH SENSITIVITY)
# ==========================================
MIN_POSSIBLE_THRESHOLD = 150   # (Lowered from 500) Allows triggering in dead-silent rooms
MAX_POSSIBLE_THRESHOLD = 8000  
NOISE_MULTIPLIER = 1.15       # (Lowered from 2.0) Voice only needs to be 20% louder than the fridge hum

# Audio Settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 512   
CHANNELS = 1

# ==========================================
# VAD Tuning (HIGH SENSITIVITY)
# ==========================================
SILENCE_TOLERANCE_SEC = 0.7   
SILENCE_THRESHOLD_BLOCKS = int(SILENCE_TOLERANCE_SEC / (BLOCK_SIZE / SAMPLE_RATE)) 
VAD_THRESHOLD = 0.3           # (Lowered from 0.5) Makes the AI easily accept soft or breathy voices
# Whisper Settings
WHISPER_MODEL_SIZE = "small.en"
USE_GPU = torch.cuda.is_available()
WHISPER_COMPUTE_TYPE = "float16" if USE_GPU else "float32"
INITIAL_PROMPT = (
    "Kitchen command context. "
    "Examples: Add two apples. Put milk into fridge. "
    "Remove two eggs. Check items. List inventory. "
    "Quantities: 1, 2, 3. Prepositions: into, inside."
)

# --- SHELF LIFE RULES ---
SHELF_LIFE_RULES = {
    "protein": 4, "seafood": 7, "dairy": 10, 
    "vegetables": 7, "fruits": 7, "grains": 14,       
    "pantry": 365, "processed": 7, "non-food": 0
}

# ------------------------
# GLOBAL STATES & ARDUINO
# ------------------------
CONVERSATION_HISTORY = collections.deque(maxlen=10)
VOICE_MODE = False
CURRENT_NOISE_LEVEL = 0
DYNAMIC_THRESHOLD = 1000
CURRENT_MIC_NAME = None

jarvis_awake = False
door_open = False
door_open_time = None
conversation_ended_time = None
tts_lock = threading.Lock()

try:
    arduino = serial.Serial("COM5", 9600, timeout=1)
    time.sleep(2)
except Exception as e:
    print("⚠️ Arduino not connected:", e)
    arduino = None

def send_to_lcd(text: str):
    if arduino:
        try:
            arduino.write((text.strip() + "\n").encode())
        except Exception:
            pass

# ------------------------
# HARDWARE THREADS (SECTION A)
# ------------------------
def listen_door_events():
    global jarvis_awake, door_open, door_open_time

    while True:
        try:
            if arduino and arduino.in_waiting:
                msg = arduino.readline().decode(errors="ignore").strip()

                if msg == "DOOR_OPEN":
                    door_open = True
                    door_open_time = time.time()
                    jarvis_awake = True
                    send_to_lcd("JARVIS: Door opened")

                elif msg == "DOOR_CLOSED":
                    door_open = False
                    door_open_time = None

        except Exception:
            pass
        time.sleep(0.1)

def door_reminder_watchdog():
    global door_open_time

    while True:
        if (
            door_open
            and not jarvis_awake                      
            and conversation_ended_time is not None   
            and time.time() - conversation_ended_time >= 30
        ):
            speak("Please close the fridge door.")
            time.sleep(30)
        time.sleep(1)


# ------------------------
# DATA LOGIC (SECTION B)
# ------------------------
def normalize_item_name(item: str, existing_names: List[str]) -> str:
    """Smart plural/singular converter to map 'strawberries' to 'strawberry'."""
    item = item.lower().strip()
    if not item: return item
    
    # Generate all possible singular/plural variations
    variations = {item}
    
    # Check for Singulars
    if item.endswith("ies"): variations.add(item[:-3] + "y")      # strawberries -> strawberry
    if item.endswith("oes"): variations.add(item[:-2] + "o")      # potatoes -> potato
    if item.endswith("s") and not item.endswith("ss"): variations.add(item[:-1]) # apples -> apple
    
    # Check for Plurals
    if item.endswith("y"): variations.add(item[:-1] + "ies")      # strawberry -> strawberries
    elif item.endswith("o"): variations.add(item + "es")          # potato -> potatoes
    else: variations.add(item + "s")                              # apple -> apples
    
    # 1. Match with existing items in fridge
    for v in variations:
        if v in existing_names:
            return v
            
    # 2. If it's completely new, force singular to keep JSON clean
    if item.endswith("ies"): return item[:-3] + "y"
    if item.endswith("s") and not item.endswith("ss"): return item[:-1]
    return item

def load_fridge() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE): return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_fridge(fridge: Dict[str, Any]) -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(fridge, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   [ERROR] Could not save JSON: {e}")

def log_waste(item: str, qty: float, unit: str):
    waste_data = []
    if os.path.exists(WASTE_FILE):
        try:
            with open(WASTE_FILE, "r", encoding="utf-8") as f: waste_data = json.load(f)
        except: pass
    
    entry = {
        "date_event": datetime.now().strftime("%Y-%m-%d"),
        "item": item,
        "qty": qty,
        "unit": unit,
        "status": "EXPIRED" 
    }
    waste_data.append(entry)
    try:
        with open(WASTE_FILE, "w", encoding="utf-8") as f:
            json.dump(waste_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   [ERROR] Could not save waste log: {e}")

def generate_waste_report(period: str) -> str:
    fridge = load_fridge()
    dirty = False
    for k, v in fridge.items():
        exp = v.get("expiry", "N/A")
        new_stat = check_status(exp)
        if v.get("status") != new_stat:
            v["status"] = new_stat
            dirty = True
    if dirty:
        save_fridge(fridge)

    # FIX: Added .date() so it ignores hours/minutes/seconds
    today = datetime.now().date() 
    
    if "week" in period.lower():
        cutoff_date = today - timedelta(days=7)
        filename = "weekly.json"
        title = "Weekly Waste"
    elif "month" in period.lower():
        cutoff_date = today - timedelta(days=30)
        filename = "monthly.json"
        title = "Monthly Waste"
    else:
        cutoff_date = today - timedelta(days=365)
        filename = "waste_report.json"
        title = "Total Waste"

    report_items = []
    if os.path.exists(WASTE_FILE):
        try:
            with open(WASTE_FILE, "r", encoding="utf-8") as f: waste_data = json.load(f)
            for entry in waste_data:
                try:
                    # FIX: Added .date()
                    event_date = datetime.strptime(entry["date_event"], "%Y-%m-%d").date() 
                    if event_date >= cutoff_date:
                        report_items.append(entry)
                except: continue
        except: pass

    for k, v in fridge.items():
        if v.get("status") == "EXPIRED":
            try:
                # FIX: Added .date()
                exp_date = datetime.strptime(v["expiry"], "%Y-%m-%d").date()
                if exp_date >= cutoff_date:
                    report_items.append({
                        "date_event": v["expiry"],
                        "item": v["item_name"],
                        "qty": v["qty"],
                        "unit": v["unit"],
                        "status": "EXPIRED"
                    })
            except: continue

    try:
        with open(os.path.join(SCRIPT_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(report_items, f, indent=2, ensure_ascii=False)
    except: pass

    if not report_items: return f"Great news. No wasted or expired food found for the {title}."

    summary = {}
    for item in report_items:
        key = f"{item['item']} ({item['unit']})"
        summary[key] = summary.get(key, 0) + item['qty']
    
    lines = [f"{format_qty(v)} {k}" for k, v in summary.items()]
    return f"{title} report generated. Total wasted or expired: {', '.join(lines)}."


def calculate_expiry(category: str) -> str:
    days = SHELF_LIFE_RULES.get(category.lower(), 7) 
    if days == 0: return "N/A"
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

def format_qty(value: float) -> str:
    if value == int(value): return str(int(value))
    return str(value)

def check_status(expiry_str: str) -> str:
    if not expiry_str or expiry_str == "N/A": return "--"
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        delta = (exp_date - today).days
        if delta < 0: return "EXPIRED" 
        elif delta <= 3: return "SOON"
        else: return "OK"
    except: return "??"

def update_fridge(action: str, item: str, qty: float, unit: str, category: str, selector: str = "oldest") -> tuple[str, str]:
    fridge = load_fridge()
    
    # --- REPLACE OLD LOGIC WITH THIS ---
    existing_names = [v['item_name'] for v in fridge.values()]
    item = normalize_item_name(item, existing_names)
    
    if not item: return "", ""
    # -----------------------------------
    
    if qty < 0: return (f"   [ERROR] Invalid qty.", f"Cannot {action} negative quantity.")
    if qty == 0: return (f"   [IGNORED] Qty 0.", f"Added nothing.")

    if not category: category = "pantry"
    if unit.lower() in ["pcs", "piece"]: unit = "pieces"

    if action == "remove" and item in ["everything", "all", "all items", "the food"]:
        if not fridge: return "   [INFO] Fridge empty.", "Fridge is empty."
        count = len(fridge)
        fridge.clear() 
        save_fridge(fridge)
        return (f"   [CLEARED] Removed {count} items.", "Removed everything.")

    term_msg, voice_msg = "", ""
    qty_str = format_qty(qty)

    if action == "add":
        expiry_date = calculate_expiry(category)
        found_key = None
        for k, v in fridge.items():
            if v['item_name'] == item and v['expiry'] == expiry_date and v['unit'] == unit:
                found_key = k
                break
        
        if found_key:
            fridge[found_key]['qty'] += qty
            fridge[found_key]['status'] = check_status(expiry_date)
            term_msg = f"   [UPDATED] {item:<15} | {format_qty(fridge[found_key]['qty'])} {unit:<10} | Exp: {expiry_date}"
            voice_msg = f"Added {qty_str} {unit} of {item} to existing batch."
        else:
            base_key = item
            new_key = base_key
            counter = 2
            while new_key in fridge:
                new_key = f"{base_key} ({counter})"
                counter += 1
            
            new_entry = {"item_name": item, "qty": qty, "unit": unit, "category": category, "expiry": expiry_date, "status": check_status(expiry_date)}
            fridge[new_key] = new_entry
            term_msg = f"   [ADDED] {item:<15} | {qty_str} {unit:<10} | Exp: {expiry_date:<10}"
            voice_msg = f"Added {qty_str} {unit} of {item}."

    elif action == "remove":
        batches = [(k, v) for k, v in fridge.items() if v['item_name'] == item]
        if not batches:
            term_msg, voice_msg = f"   [ERROR] No {item} found.", f"You don't have any {item}."
        else:
            reverse_sort = True if (selector and selector.lower() in ["newest", "latest", "new", "fresh"]) else False
            sort_desc = "Newest" if reverse_sort else "Oldest"
            batches.sort(key=lambda x: x[1].get('expiry', '9999-99-99'), reverse=reverse_sort)
            
            qty_to_remove = qty
            removed_log = []
            
            for k, v in batches:
                if qty_to_remove <= 0: break
                available = v['qty']
                deduct = min(available, qty_to_remove)
                
                v['qty'] -= deduct
                qty_to_remove -= deduct
                
                if check_status(v.get('expiry', 'N/A')) == "EXPIRED":
                    log_waste(item, deduct, v['unit']) 

                removed_log.append(f"{deduct} (Exp: {v.get('expiry')})")
                if v['qty'] <= 0: fridge.pop(k)
            
            if qty_to_remove > 0:
                term_msg = f"   [PARTIAL] Removed {qty - qty_to_remove} {item} ({sort_desc}). Ran out."
                voice_msg = f"Removed {format_qty(qty - qty_to_remove)} {item}. That is all you had."
            else:
                term_msg = f"   [REMOVED] {item:<13} | {qty_str} total | Mode: {sort_desc}"
                voice_msg = f"Removed {qty_str} {item}."

    save_fridge(fridge)
    return term_msg, voice_msg

def get_list_outputs() -> tuple[str, str]:
    fridge = load_fridge()
    if not fridge: return "   [EMPTY] Fridge is empty.", "The fridge is empty."
    term_lines = ["\n📦 INVENTORY:", "-"*65, f"{'ITEM':<15} | {'QTY':<8} | {'EXPIRY':<12} | {'STATUS'}", "-"*65]
    voice_parts = []
    
    sorted_items = sorted(fridge.values(), key=lambda x: (x['item_name'], x.get('expiry', '9999')))
    for v in sorted_items:
        term_lines.append(f"{v['item_name']:<15} | {format_qty(v['qty'])} {v['unit']:<5} | {v.get('expiry','N/A'):<12} | {v.get('status','OK')}")
    
    item_sums = {}
    for v in fridge.values():
        k = f"{v['item_name']} {v['unit']}"
        item_sums[k] = item_sums.get(k, 0) + v['qty']
    
    for k, total in item_sums.items():
        voice_parts.append(f"{format_qty(total)} {k}")

    return "\n".join(term_lines) + "\n", "Inventory: " + ", ".join(voice_parts) + "."

def get_lookup_output(item: str) -> tuple[str, str]:
    fridge = load_fridge()
    
    # --- REPLACE OLD LOGIC WITH THIS ---
    item = normalize_item_name(item, []) # Get base singular form
    search_terms = {item, item + "s", item[:-1] if item.endswith("s") else item,
                    item[:-3] + "y" if item.endswith("ies") else item,
                    item[:-1] + "ies" if item.endswith("y") else item}
                    
    matches = [v for v in fridge.values() if v['item_name'].lower() in search_terms]
    # -----------------------------------
            
    if not matches: return f"   [LOOKUP] No {item} found.", f"You don't have any {item}."
    matches.sort(key=lambda x: x.get('expiry', '9999'))
    
    term_lines = [f"\n🔍 RESULT: {item}"]
    voice_parts = []
    for m in matches:
        term_lines.append(f"   - {format_qty(m['qty'])} {m['unit']} | Exp: {m['expiry']} | Stat: {m.get('status','OK')}")
        voice_parts.append(f"{format_qty(m['qty'])} {m['unit']}")

    return "\n".join(term_lines), f"You have: " + " and ".join(voice_parts) + "."


# ------------------------
# SPEECH / AUDIO
# ------------------------
def clean_stt_text(text: str) -> str:
    text_lower = text.lower()
    text_lower = re.sub(r'\badd to\b', 'add two', text_lower)
    text_lower = re.sub(r'\bremove to\b', 'remove two', text_lower)
    return text_lower

def get_quantity_from_speech(text: str) -> int:
    text = text.lower()
    digits = re.findall(r"\b(\d+)\b", text)
    if digits: return int(digits[0])
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    for word, num in word_to_num.items():
        if f" {word} " in f" {text} ": return num
    return 1 

# --- PRE-LOAD VOICE ENGINE HERE ---
try:
    jarvis_voice = pyttsx3.init()
    jarvis_voice.setProperty("rate", 190) 
    jarvis_voice.setProperty("volume", 0.9)
except Exception as e:
    print(f"Failed to load TTS engine: {e}")
    jarvis_voice = None

def speak(text: str) -> None:
    text = text.replace(" pcs", " pieces")
    with tts_lock:
        send_to_lcd(f"JARVIS: {text}")
        if jarvis_voice:
            try:
                jarvis_voice.say(text)
                jarvis_voice.runAndWait()
            except Exception as e:
                print(f"   [TTS Error]: {e}")

def play_beep():
    """Plays a quick, clean beep sound to indicate Jarvis is listening."""
    try:
        fs = 16000
        # Generate a 0.15 second tone at 880Hz (A5 note)
        t = np.linspace(0, 0.15, int(fs * 0.15), False)
        beep = np.sin(880 * t * 2 * np.pi) * 0.3 
        
        sd.play(beep, samplerate=fs)
        sd.wait() # Wait for the beep to finish
        
        # Tiny delay so the mic doesn't hear the beep echoing in the room
        time.sleep(0.2) 
    except Exception:
        pass
                
    # NEW: Forces the script to pause for 0.4 seconds so the mic 
    # doesn't pick up the speaker echo or hardware 'clicks'
    time.sleep(0.4)


# STT INIT
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda" if USE_GPU else "cpu", compute_type=WHISPER_COMPUTE_TYPE)
torch.set_num_threads(1)
vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def listen_dynamic():
    global CURRENT_NOISE_LEVEL, DYNAMIC_THRESHOLD, CURRENT_MIC_NAME
    try:
        current_mic = sd.query_devices(kind='input')['name']
        if CURRENT_MIC_NAME is not None and current_mic != CURRENT_MIC_NAME: pass
        CURRENT_MIC_NAME = current_mic
    except Exception: pass

    q = queue.Queue()
    def callback(indata, frames, time_, status): q.put(indata.copy())

    pad_blocks = max(1, int((0.5 * SAMPLE_RATE) / BLOCK_SIZE))
    pre_buffer = collections.deque(maxlen=pad_blocks)
    audio_blocks, speech_started, silence_blocks = [], False, 0
    noise_buffer = collections.deque(maxlen=20) 
    last_ui_update = 0
    
    try:
        with sd.InputStream(device=None, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK_SIZE, dtype="float32", callback=callback):
            vad_iterator = VADIterator(vad_model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=100)
            start_speech_time = 0 

            while True:
                try: data = q.get(timeout=0.5) 
                except queue.Empty: continue

                pcm16 = (data[:, 0] * 32767).astype(np.int16)
                if len(pcm16) != 512: pcm16 = np.pad(pcm16, (0, 512 - len(pcm16)))

                current_peak = np.max(np.abs(pcm16))
                noise_buffer.append(current_peak)
                avg_noise = int(sum(noise_buffer) / len(noise_buffer))
                target_thresh = min(MAX_POSSIBLE_THRESHOLD, max(MIN_POSSIBLE_THRESHOLD, int(avg_noise * NOISE_MULTIPLIER)))
                DYNAMIC_THRESHOLD, CURRENT_NOISE_LEVEL = target_thresh, avg_noise

                if time.time() - last_ui_update > 0.1:
                    rec_icon = "🔴 REC" if speech_started else "👂 Listen"
                    sys.stdout.write(f"\r   [🟢] {rec_icon} | Noise:{CURRENT_NOISE_LEVEL:<4} | Thresh:{DYNAMIC_THRESHOLD:<4} | Vol:{current_peak:<5}   ")
                    sys.stdout.flush()
                    last_ui_update = time.time()

                is_loud_enough = current_peak > DYNAMIC_THRESHOLD
                speech_dict = vad_iterator(torch.from_numpy(pcm16), SAMPLE_RATE) if (is_loud_enough or speech_started) else {}

                if speech_dict:
                    if not speech_started:
                        speech_started = True
                        if jarvis_awake:
                            send_to_lcd("Listening...")
                        start_speech_time = time.time()
                        audio_blocks.extend(list(pre_buffer)) 
                    audio_blocks.append(pcm16)
                    silence_blocks = 0
                else:
                    if not speech_started: pre_buffer.append(pcm16)
                    else:
                        silence_blocks += 1
                        audio_blocks.append(pcm16)
                        if silence_blocks >= SILENCE_THRESHOLD_BLOCKS or (time.time() - start_speech_time > 15): break 
                            
    except Exception as e:
        time.sleep(1)
        return ""

    vad_iterator.reset_states()
    if not audio_blocks: return ""
    
    full_audio = np.concatenate(audio_blocks).astype(np.int16)
    if len(full_audio) < 4000: return ""

    sys.stdout.write(f"\r   📝 Transcribing...                            ")
    sys.stdout.flush()
    send_to_lcd("Transcribing...") # <--- ADD THIS LINE

    audio_float = full_audio.astype(np.float32) / 32768.0

    audio_float = full_audio.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio_float, beam_size=5, language="en", initial_prompt=INITIAL_PROMPT)
    return " ".join([s.text for s in segments]).strip()

# ------------------------
# EXECUTION
# ------------------------
def ask_llm(user_input: str, context: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    history_str = "\n".join(CONVERSATION_HISTORY)
    prompt = f"Current Date: {today}\n\n### CONTEXT\n{context}\n\n### HISTORY\n{history_str}\n\n### USER INPUT\nUser: \"{user_input}\""
    
    try:
        resp = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False}, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["message"]["content"]
            clean = content.replace("```json", "").replace("```", "").strip()
            s, e = clean.find('['), clean.rfind(']') + 1
            return clean[s:e] if s != -1 and e != -1 else "[]"
    except: pass
    return "[]"

def execute(actions: List[Dict], user_text: str) -> None:
    print("\n" + "="*60)
    print(f"👤 USER COMMAND: {user_text}")
    print("-" * 60)
    CONVERSATION_HISTORY.append(f"User: {user_text}")

    if not actions:
        sys.stdout.flush()
        speak("I didn't catch that.")
        return

    # Section A Override Logic
    forced_qty = get_quantity_from_speech(user_text)

    terminal_buffer, voice_parts = [], []

    for a in actions:
        act = a.get("action", "none")
        item = a.get("item", "").strip()
        
        # Override AI qty if regex found a bigger specific number
        ai_val = float(a.get("quantity", a.get("count", a.get("qty", 1))))
        qty = forced_qty if (ai_val == 1 and forced_qty > 1) else ai_val
        
        unit = a.get("unit", "pieces") 
        cat = a.get("category", "pantry")
        response = a.get("response", "")
        period = a.get("period", "weekly")
        selector = a.get("selector", "oldest")

        t_out, v_out = "", ""

        if act == "list":
            t_out, v_out = get_list_outputs()
        elif act in ["add", "remove"]:
            t_out, v_out = update_fridge(act, item, qty, unit, cat, selector)
        elif act == "lookup":
            t_out, v_out = get_lookup_output(item)
        elif act == "summarize":
            v_out = generate_waste_report(period)
            t_out = f"   [SUMMARY] {v_out}"
        elif act == "chitchat":
            v_out = response if response else "I am online."
        elif act == "none": continue

        if t_out: terminal_buffer.append(t_out)
        if v_out: voice_parts.append(v_out)

    if terminal_buffer: print("\n".join(terminal_buffer)) 
    
    full_voice = ". ".join(voice_parts) + "."
    if voice_parts:
        print(f"\n🤖 JARVIS: {full_voice}")
        speak(full_voice)
        if len(full_voice) > 200 and "inventory" in full_voice.lower():
            CONVERSATION_HISTORY.append("Jarvis: [Inventory List Output]")
        else:
            CONVERSATION_HISTORY.append(f"Jarvis: {full_voice}")
            
    print("="*60 + "\n")
    sys.stdout.flush() 

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    print(f"--- Jarvis (Merged Hardware + STT2) ---")
    
    # Start Hardware Background Threads
    threading.Thread(target=listen_door_events, daemon=True).start()
    threading.Thread(target=door_reminder_watchdog, daemon=True).start()

    try:
        CURRENT_MIC_NAME = sd.query_devices(kind='input')['name']
        print(f"   [MIC] Connected to: {CURRENT_MIC_NAME}")
    except Exception:
        pass

    speak("System Initialized.")
    
    try:
        while True:
            # 1. Gather Input (Fixed for Windows CMD)
            text = ""
            if VOICE_MODE:
                text = listen_dynamic()
            else:
                try: 
                    # Combining this into one line prevents Windows CMD from freezing
                    text = input("\n   👉 Type here: ")
                except EOFError: 
                    break

            if not text: continue
            
            text_clean = clean_stt_text(text)
            text_lower = text_clean.lower()
            send_to_lcd(f"YOU: {text_clean}")
            # 2. Mode Switching Commands (Restored)
            if any(x in text_lower for x in ["switch to text", "text mode"]):
                VOICE_MODE = False
                print("\n   [SYSTEM] Switched to TEXT MODE.")
                send_to_lcd("Mode: TEXT")
                speak("Switched to text mode.")
                continue
            
            if any(x in text_lower for x in ["switch to voice", "voice mode"]):
                VOICE_MODE = True
                print("\n   [SYSTEM] Switched to VOICE MODE.")
                send_to_lcd("Mode: VOICE")
                speak("Switched to voice mode.")
                continue

            # ==========================================
            # SECTION A WAKE WORD LOGIC
            # ==========================================
            # 3. ONLY require the wake word if using the Microphone!
            if VOICE_MODE and not jarvis_awake:
                # Reverted back to just "hey" (and "hay" for STT spelling errors)
                wake_phrases = ["hey", "hay"]
                
                # Using \b (word boundary) ensures words like "they" don't trigger it
                if any(re.search(rf'\b{phrase}\b', text_lower) for phrase in wake_phrases):
                    jarvis_awake = True
                    conversation_ended_time = None
                    print("\n   [🗣️] Jarvis is listening...")
                    send_to_lcd("Listening...")
                    speak("I'm ready to listen.") 
                
                # Always restart the loop if asleep, forcing the user to wait for the greeting
                continue

            # Sleep / Quit Logic
            if any(x in text_lower for x in ["goodbye", "exit", "go to sleep"]):
                speak("Thank you. Have a nice day.")
                jarvis_awake = False
                conversation_ended_time = time.time()
                continue
            # ==========================================

            if VOICE_MODE: print(f"\n   ⏳ Processing...")
            send_to_lcd("Processing...")
            
            # --- NEW STRICT LLM INSTRUCTIONS ---
            system_instructions = (
                "You are a smart fridge. Reply ONLY in JSON array format. "
                "Valid actions: 'add', 'remove', 'list', 'lookup', 'summarize', 'chitchat'. "
                "If the user asks about waste, expired food, or a report, output exactly: "
                "[{\"action\": \"summarize\", \"period\": \"weekly\"}] (period: weekly, monthly, or all). "
                "Do not include plain text outside the JSON."
            )
            json_response = ask_llm(text_clean, system_instructions) 
            # -----------------------------------
            
            try: execute(json.loads(json_response), text_clean)
            except Exception as e: 
                print(f"Exec Error: {e}")
                speak("Processing Error.")

    except KeyboardInterrupt:
        print("\nSystem terminated")
        os._exit(0)
